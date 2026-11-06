"""`ai.workers.Worker`: a coding-agent CLI that goes where a model goes.

A worker is ONE conversation with one coding-agent CLI, such as Codex. It is
a pydantic model: its fields are its settings, the CLI's `session_id` and
the conversation so far (`messages`). Dump it, store it, and validate it
again later; it continues the same conversation.

It follows the model protocol: `stream(messages, tools=, output_type=)`
yields the events of one assistant message, so `ai.stream`,
`ai.experimental_generate` and `ai.Agent.run` take a worker where they take
a model. The CLI runs its own loop and its own tools; those come back as
`BuiltinTool*` events, so an `ai.Agent` does not run them again. The tools
of an `ai.Agent` reach the CLI through the MCP bridge (see `_bridge`) and
come back as normal tool calls, for the agent to run.

A worker has state, a model does not. `stream()` still receives the whole
conversation on every call. The worker compares it with `messages` and
sends the CLI only what is new (see `_plan`).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any, ClassVar, Literal, Self

import pydantic
from pydantic import TypeAdapter

from ..agents import hooks
from ..agents import runtime as runtime_
from ..types import events as events_
from ..types import messages as messages_
from ..types import tools as tools_
from ..types import usage as usage_
from . import _bridge
from . import _errors as errors
from ._adapters import base

Effort = Literal["low", "medium", "high", "max"]

BATCH_WINDOW = 0.05
"""Seconds to wait for more bridge calls before handing a batch to the
caller. A CLI that calls tools in parallel sends them close together."""

OUTPUT_RETRIES = 2
"""How many times to ask again for output that does not validate."""

OUTPUT_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON value matching this schema, "
    "no prose, no code fences:\n"
)


@dataclasses.dataclass
class _Call:
    """A call from the CLI to an `ai` tool, through the bridge."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    answer: asyncio.Future[tuple[str, bool]]


@dataclasses.dataclass
class _Failed:
    error: BaseException


@dataclasses.dataclass
class _Plan:
    """What one `stream()` call must do, from the messages it was given."""

    seed: list[messages_.Message] = dataclasses.field(default_factory=list)
    instructions: str | None = None
    results: list[messages_.ToolResultPart] = dataclasses.field(
        default_factory=list
    )
    system: list[str] = dataclasses.field(default_factory=list)
    """System text that arrived after the conversation began."""
    prompt: list[messages_.TextPart | messages_.FilePart] = dataclasses.field(
        default_factory=list
    )
    transcript: list[messages_.Message] = dataclasses.field(
        default_factory=list
    )
    """The transcript once this call succeeds, before its answer."""


@dataclasses.dataclass
class _Runtime:
    """What lives only while a worker is open. Never copied, never dumped."""

    adapter: base.Adapter
    bridge: _bridge.Bridge
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    in_session: bool = False
    # The running turn: its pump task and the queue it fills.
    queue: asyncio.Queue[Any] | None = None
    pump: asyncio.Task[None] | None = None
    # Bridge calls handed to the caller, waiting for their results.
    pending: dict[str, _Call] = dataclasses.field(default_factory=dict)
    # CLI tool ids that are bridge calls, hidden from the stream.
    hidden: set[str] = dataclasses.field(default_factory=set)
    # The context of the latest `stream()` call: approvals ask in it.
    context: contextvars.Context | None = None


class Worker(pydantic.BaseModel):
    """Base class of the workers, such as `Codex`.

    `Worker.model_validate(data)` builds the kind that `data["kind"]` names.
    """

    kind: str
    model: str | None = None
    """The CLI's model. None: the CLI's own default."""
    cwd: str = pydantic.Field(default_factory=os.getcwd)
    """Where the CLI runs and keeps its sessions."""
    env: dict[str, str] = {}
    """Added to the CLI's environment: keys, base URLs, config dirs."""
    executable: str | None = None
    effort: Effort | None = None
    mcp_servers: dict[str, dict[str, Any]] = {}
    """MCP servers for the CLI.

    `{"name": {"command": ..., "args": [...]}}` or `{"name": {"url": ...}}`.
    """
    approve: bool = False
    """Ask before each of the CLI's own tool calls, through `ai.hook`.

    False: the CLI's own permission settings decide (`allowed_tools`,
    Codex's `sandbox`). True: each call becomes a `ToolApproval` hook, which
    `ai.resolve_hook` and UIs answer as they answer any tool approval. Needs
    an `ai.Agent.run()`, where hooks live.
    """
    session_id: str | None = None
    """The CLI's own id for the conversation. Set it to resume one."""
    fork: bool = False
    """With `session_id`: fork that conversation natively on first use."""
    messages: list[messages_.Message] = []
    """The conversation so far. Send it back with the next user message."""

    _kinds: ClassVar[dict[str, type[Worker]]] = {}
    _runtime: _Runtime | None = pydantic.PrivateAttr(default=None)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201
        super().__pydantic_init_subclass__(**kwargs)
        kind = cls.model_fields["kind"].default
        if isinstance(kind, str):
            Worker._kinds[kind] = cls

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _of_kind(
        cls, data: Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Any:
        if cls is Worker and isinstance(data, dict):
            kind = Worker._kinds.get(str(data.get("kind")))
            if kind is None:
                raise ValueError(f"no worker of kind {data.get('kind')!r}")
            return kind.model_validate(data)
        return handler(data)

    def __copy__(self) -> Self:
        # A copy is a new worker: it opens a CLI of its own.
        copied = super().__copy__()
        copied._runtime = None
        return copied

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = super().__deepcopy__(memo)
        copied._runtime = None
        return copied

    # -- subclass point -------------------------------------------------------

    def _make_adapter(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        approve: base.Approve | None,
    ) -> base.Adapter:
        raise NotImplementedError

    # -- ModelLike ------------------------------------------------------------

    @property
    def id(self) -> str:
        return self.model or "default"

    @property
    def provider_name(self) -> str:
        return self.kind

    async def stream(
        self,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
    ) -> AsyncGenerator[events_.Event]:
        """Answer `messages` with one assistant message.

        Use `ai.stream(worker, messages)`, as with a model. The message ends
        in one of two ways:

        - `finish_reason="tool_call"`: the CLI called `ai` tools from
          `tools=`. The message holds them as `ToolCallPart`s. Run them and
          call again with the results; the CLI waits until then.
        - Anything else: the CLI's turn is over.

        `params` is ignored: the CLI chooses its own.
        """
        rt = await self._open()
        await rt.lock.acquire()
        paused = False
        try:
            if self.approve:
                try:
                    hooks.get_hook_registry()
                    runtime_.get_runtime()
                except LookupError:
                    raise errors.WorkerError(
                        "approve=True asks through ai.hook, which needs a "
                        "run: use the worker in ai.Agent.run()"
                    ) from None
            rt.context = contextvars.copy_context()
            plan = self._plan(messages)
            rt.bridge.set_tools(
                [t for t in tools or () if t.kind == "function"]
            )
            if rt.pending:
                if plan.results:
                    self._answer(plan.results)
                else:
                    await self._abandon()
            if plan.prompt:
                if rt.in_session and plan.system:
                    text = "\n\n".join(plan.system) + "\n\n"
                    plan.prompt.insert(0, messages_.TextPart(text=text))
                if rt.queue is not None:
                    # Tool results came with a new user message. The CLI is
                    # still in its turn: the message is guidance for it.
                    await rt.adapter.steer(base.text_of(plan.prompt))
                else:
                    await self._ensure_session(plan)
                    self._start(plan.prompt, output_type)
            if rt.queue is None:
                raise ValueError(
                    "stream(): no new user message or tool result to send"
                )

            stamp = messages_.Message(role="assistant", parts=[])
            hydrator = events_.MessageHydrator()

            def out(event: events_.Event) -> events_.Event:
                event = event.model_copy(
                    update={
                        "message": messages_.Message(
                            id=stamp.id, role="assistant", parts=[]
                        )
                    }
                )
                hydrator.feed(event)
                return event

            yield out(events_.StreamStart())
            # Text after the last CLI tool call: the answer, for output_type.
            answer: list[str] = []
            while True:
                item = await rt.queue.get()
                if isinstance(item, _Failed):
                    self._end_turn()
                    raise item.error
                if isinstance(item, _Call):
                    calls = [item]
                    with contextlib.suppress(TimeoutError):
                        async with asyncio.timeout(BATCH_WINDOW):
                            while True:
                                more = await rt.queue.get()
                                if isinstance(more, _Call):
                                    calls.append(more)
                                else:
                                    # Not ours to handle here: put it back.
                                    rt.queue.put_nowait(more)
                                    break
                    for call in calls:
                        rt.pending[call.tool_call_id] = call
                        part = messages_.ToolCallPart(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.name,
                            tool_args=json.dumps(call.arguments),
                        )
                        yield out(
                            events_.ToolStart(
                                tool_call_id=part.tool_call_id,
                                tool_name=part.tool_name,
                            )
                        )
                        yield out(
                            events_.ToolDelta(
                                tool_call_id=part.tool_call_id,
                                chunk=part.tool_args,
                            )
                        )
                        yield out(
                            events_.ToolEnd(
                                tool_call_id=part.tool_call_id, tool_call=part
                            )
                        )
                    paused = True
                    yield out(
                        events_.StreamEnd(
                            finish_reason="tool_call",
                            provider_metadata=self._metadata(None),
                        )
                    )
                    break
                if isinstance(item, events_.StreamEnd):
                    self._end_turn()
                    usage = item.usage
                    if output_type is not None:
                        text, usage = await self._output(
                            "".join(answer), output_type, usage
                        )
                        yield out(events_.TextStart(block_id="output"))
                        yield out(
                            events_.TextDelta(chunk=text, block_id="output")
                        )
                        yield out(events_.TextEnd(block_id="output"))
                    finish = item.finish_reason
                    yield out(
                        events_.StreamEnd(
                            finish_reason=(
                                "other" if finish == "cancelled" else finish
                            ),
                            usage=usage,
                            provider_metadata={
                                **(item.provider_metadata or {}),
                                **self._metadata(finish),
                            },
                        )
                    )
                    break
                for event in self._visible(item):
                    if output_type is not None and isinstance(
                        event,
                        events_.TextStart | events_.TextDelta | events_.TextEnd,
                    ):
                        # Narration is not the output. Keep only the text
                        # after the last tool call, and send it once parsed.
                        if isinstance(event, events_.TextDelta):
                            answer.append(event.chunk)
                        continue
                    if isinstance(event, events_.BuiltinToolStart):
                        answer.clear()
                    yield out(event)
            # Committed only now: a call that failed can be sent again.
            self.messages = [*plan.transcript, hydrator.message]
        except (GeneratorExit, asyncio.CancelledError):
            if not paused:
                # Nobody reads this turn any more: end it in the CLI too.
                await self._abandon()
            raise
        finally:
            rt.lock.release()

    # -- the conversation -----------------------------------------------------

    def _plan(self, messages: list[messages_.Message]) -> _Plan:
        """Compare `messages` with the transcript; return what is new."""
        known = [m.id for m in self.messages]
        given = [m.id for m in messages]
        plan = _Plan()
        system = [m.text for m in messages if m.role == "system"]
        plan.instructions = "\n\n".join(system) or None
        if known:
            if given[: len(known)] != known:
                raise errors.TranscriptMismatchError(
                    f"{self.kind}: these messages do not continue this "
                    "worker's conversation; start a new worker"
                )
            new = messages[len(known) :]
            plan.transcript = list(self.messages)
        elif self.session_id is not None:
            # A stored conversation the worker has not loaded: the CLI holds
            # the past, so everything given is new. `load()` reads the past.
            new = list(messages)
        else:
            # A new conversation. What comes before the last user turn is
            # its past; the CLI is seeded with it before the first turn.
            cut = len(messages)
            while cut and messages[cut - 1].role in ("user", "system", "tool"):
                cut -= 1
            plan.seed = [m for m in messages[:cut] if m.role != "system"]
            new = messages[cut:]
            plan.transcript = list(messages[:cut])
        for message in new:
            if message.role == "tool":
                plan.results.extend(message.tool_results)
            elif message.role == "system":
                plan.system.append(message.text)
            elif message.role == "user":
                plan.prompt.extend(
                    p
                    for p in message.parts
                    if isinstance(p, messages_.TextPart | messages_.FilePart)
                )
            else:
                raise errors.TranscriptMismatchError(
                    f"{self.kind}: a {message.role} message the worker did "
                    "not write; only user, system and tool messages can "
                    "follow its last answer (load() reads a stored past)"
                )
        plan.transcript.extend(new)
        return plan

    async def _ensure_session(self, plan: _Plan) -> None:
        rt = self._require_runtime()
        if rt.in_session:
            return
        self.session_id = await rt.adapter.open(
            session_id=self.session_id,
            fork=self.fork,
            seed=plan.seed if self.session_id is None and plan.seed else None,
            instructions=plan.instructions,
        )
        self.fork = False
        rt.in_session = True

    def _start(
        self,
        prompt: list[messages_.TextPart | messages_.FilePart],
        output_type: type[pydantic.BaseModel] | None,
    ) -> None:
        rt = self._require_runtime()
        if output_type is not None:
            schema = json.dumps(TypeAdapter(output_type).json_schema())
            prompt = [
                *prompt,
                messages_.TextPart(text=OUTPUT_INSTRUCTION + schema),
            ]
        queue: asyncio.Queue[Any] = asyncio.Queue()
        adapter = rt.adapter

        async def pump() -> None:
            try:
                async for event in adapter.turn(prompt):
                    queue.put_nowait(event)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                queue.put_nowait(_Failed(exc))

        rt.queue = queue
        rt.pump = asyncio.create_task(pump())

    def _end_turn(self) -> None:
        rt = self._require_runtime()
        rt.queue = None
        rt.pump = None
        rt.hidden.clear()

    def _answer(self, results: list[messages_.ToolResultPart]) -> None:
        rt = self._require_runtime()
        for part in results:
            call = rt.pending.pop(part.tool_call_id, None)
            if call is None or call.answer.done():
                continue
            value = part.model_input
            if value is messages_.MODEL_INPUT_UNSET:
                value = part.result
            text = value if isinstance(value, str) else _dump(value)
            call.answer.set_result((text, part.result_kind == "error"))
        for call in rt.pending.values():
            # A call the caller did not answer: tell the CLI, do not hang.
            if not call.answer.done():
                call.answer.set_result(("no result was given", True))
        rt.pending.clear()

    async def _abandon(self) -> None:
        """End the running turn: the caller moved on without it."""
        rt = self._runtime
        if rt is None:
            return
        for call in rt.pending.values():
            if not call.answer.done():
                call.answer.set_result(("cancelled by the caller", True))
        rt.pending.clear()
        pump = rt.pump
        if pump is None:
            return
        with contextlib.suppress(Exception):
            await rt.adapter.interrupt()
        pump.cancel()
        with contextlib.suppress(BaseException):
            await pump
        self._end_turn()

    async def _on_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        """Hand a bridge call to the caller, and wait for the result."""
        rt = self._runtime
        if rt is None or rt.queue is None:
            return ("no turn is running", True)
        answer: asyncio.Future[tuple[str, bool]] = (
            asyncio.get_running_loop().create_future()
        )
        rt.queue.put_nowait(
            _Call(messages_.generate_id("call"), name, arguments, answer)
        )
        return await answer

    async def _approve(
        self, tool_call_id: str, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[bool, str]:
        """Ask about one of the CLI's own tool calls, through `ai.hook`."""
        if self._is_bridge_tool(tool_name):
            # An `ai` tool: the agent gates it itself, if it is gated.
            return True, ""
        rt = self._runtime
        if rt is None or rt.context is None:
            return False, "no run to ask in; denied"

        async def ask() -> tuple[bool, str]:
            approval = await hooks.hook(
                f"approve_{tool_call_id}",
                payload=hooks.ToolApproval,
                metadata={"tool": tool_name, "kwargs": tool_input},
                tool_call_id=tool_call_id,
            )
            return approval.granted, approval.reason or "denied"

        # In the caller's context: that is where the run's hooks live.
        task = asyncio.create_task(ask(), context=rt.context)
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return False, "the approval was cancelled"
        except Exception as exc:
            return False, f"approval failed ({type(exc).__name__}); denied"

    def _is_bridge_tool(self, tool_name: str) -> bool:
        """Whether a tool the CLI reports is a bridge call.

        Those reach the caller as `ToolCallPart`s, so the CLI's own report
        of them is hidden. Claude Code names an MCP tool
        `mcp__<server>__<name>`; codex uses the bare name, or the server's
        name when the tool name is missing (measured: codex 0.160). How pi
        names them is not measured yet: the forms it may use are accepted.
        """
        rt = self._runtime
        if rt is None:
            return False
        server = _bridge.SERVER_NAME
        return tool_name == server or any(
            tool_name
            in (n, f"mcp__{server}__{n}", f"{server}_{n}", f"{server}__{n}")
            for n in rt.bridge.names
        )

    def _visible(self, event: events_.Event) -> list[events_.Event]:
        """Drop StreamStart and the CLI's own report of bridge calls."""
        rt = self._require_runtime()
        match event:
            case events_.StreamStart():
                return []  # a worker message has its own
            case events_.BuiltinToolStart():
                if self._is_bridge_tool(event.tool_name):
                    rt.hidden.add(event.tool_call_id)
                    return []
            case events_.BuiltinToolEnd():
                if event.tool_call_id in rt.hidden or self._is_bridge_tool(
                    event.tool_call.tool_name
                ):
                    rt.hidden.add(event.tool_call_id)
                    return []
            case events_.BuiltinToolDelta() | events_.BuiltinToolResult():
                if event.tool_call_id in rt.hidden:
                    return []
        return [event]

    async def _output(
        self,
        text: str,
        output_type: type[pydantic.BaseModel],
        usage: usage_.Usage | None,
    ) -> tuple[str, usage_.Usage | None]:
        """Parse the answer, asking again when it does not validate."""
        adapter = self._require_runtime().adapter
        parsed, error = _parse(text, output_type)
        attempt = 0
        while error is not None and attempt < OUTPUT_RETRIES:
            attempt += 1
            retry = messages_.TextPart(
                text=f"Your previous reply failed validation: {error}\n"
                "Reply again with ONLY the corrected JSON."
            )
            chunks: list[str] = []
            async for event in adapter.turn([retry]):
                if isinstance(event, events_.BuiltinToolStart):
                    chunks.clear()
                elif isinstance(event, events_.TextDelta):
                    chunks.append(event.chunk)
                elif isinstance(event, events_.StreamEnd) and event.usage:
                    usage = usage + event.usage if usage else event.usage
            text = "".join(chunks)
            parsed, error = _parse(text, output_type)
        return (parsed.model_dump_json() if parsed else text), usage

    def _metadata(self, finish: str | None) -> dict[str, Any]:
        meta: dict[str, Any] = {"kind": self.kind}
        if self.session_id:
            meta["session_id"] = self.session_id
        if finish is not None:
            meta["finish_reason"] = finish
        return {"worker": meta}

    # -- lifecycle ------------------------------------------------------------

    async def _open(self) -> _Runtime:
        if self._runtime is not None:
            return self._runtime
        bridge = _bridge.Bridge(self._on_call)
        await bridge.start()
        servers = {
            **self.mcp_servers,
            _bridge.SERVER_NAME: {"type": "http", "url": bridge.url},
        }
        adapter = self._make_adapter(
            servers, self._approve if self.approve else None
        )
        try:
            await adapter.start()
        except BaseException:
            await bridge.close()
            raise
        self._runtime = _Runtime(adapter=adapter, bridge=bridge)
        return self._runtime

    async def close(self) -> None:
        """End the turn, the CLI and the bridge. The fields stay as they are.

        A closed worker opens again on its next use, and resumes its
        conversation.
        """
        rt = self._runtime
        if rt is None:
            return
        await self._abandon()
        self._runtime = None
        try:
            await rt.adapter.close()
        finally:
            await rt.bridge.close()

    async def __aenter__(self) -> Self:
        await self._open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _require_runtime(self) -> _Runtime:
        if self._runtime is None:
            raise errors.WorkerError(f"{self.kind} worker is not open")
        return self._runtime

    @property
    def version(self) -> str | None:
        """What the CLI's `--version` printed, once the worker is open."""
        return self._runtime.adapter.version if self._runtime else None

    # -- what only a CLI has --------------------------------------------------

    async def load(self) -> None:
        """Read the stored conversation `session_id` names into `messages`.

        Also after a person worked in the CLI's TUI (see `tui()`).
        """
        if self.session_id is None:
            raise errors.WorkerError(f"{self.kind}: no session_id to load")
        rt = await self._open()
        self.messages = await rt.adapter.history(self.session_id)

    async def steer(self, text: str) -> None:
        """Send guidance to the turn that runs now; the agent acts on it.

        For use from another task while `stream()` runs. The text is not
        added to `messages`; `load()` reads it back from the CLI.
        """
        rt = self._require_runtime()
        if rt.queue is None:
            raise errors.WorkerError(f"{self.kind}: no turn is running")
        await rt.adapter.steer(text)

    def tui_command(self) -> tuple[list[str], dict[str, str]]:
        """Return the CLI's own TUI command for this conversation.

        Run it in `cwd`, with the env added to yours::

            argv, env = worker.tui_command()
            subprocess.run(argv, cwd=worker.cwd, env={**os.environ, **env})

        A worker with no conversation yet gets a new one, when the CLI lets
        us choose its id (Claude Code, Pi). Close the worker first: one
        writer per conversation.
        """
        if self.fork:
            raise errors.WorkerError(
                f"{self.kind}: run a turn first; the fork happens on first use"
            )
        rt = self._runtime
        adapter = (
            rt.adapter if rt else self._make_adapter(self.mcp_servers, None)
        )
        argv, env, minted = adapter.tui_command(self.session_id)
        if self.session_id is None:
            if minted is None:
                raise errors.WorkerError(
                    f"{self.kind} picks its own session id: run a turn "
                    "first, or pass session_id="
                )
            self.session_id = minted
        return argv, env

    async def tui(self) -> int:
        """Hand the conversation to a person, in this terminal.

        Runs the CLI's TUI on it and returns its exit code once the person
        quits. The worker then reads the conversation back (`load()`) and
        goes on headless.
        """
        await self.close()
        argv, env = self.tui_command()
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.cwd, env={**os.environ, **env}
        )
        code = await proc.wait()
        await self.load()
        return code

    @classmethod
    async def list_sessions(cls, **fields: Any) -> list[base.SessionInfo]:
        """Return the conversations the CLI has stored for a directory.

        Takes the worker's fields: `cwd=`, `env=`, `executable=`.
        """
        worker = cls(**fields)
        adapter = worker._make_adapter({}, None)
        await adapter.start()
        try:
            return await adapter.list_sessions()
        finally:
            await adapter.close()


def _parse(
    text: str, output_type: type[pydantic.BaseModel]
) -> tuple[pydantic.BaseModel | None, Exception | None]:
    try:
        return output_type.model_validate_json(_extract_json(text)), None
    except Exception as exc:
        return None, exc


def _extract_json(text: str) -> str:
    """Best effort: the model was told "only JSON", but fences happen."""
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate = parts[1].removeprefix("json").strip()
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if 0 <= start < end:
            sliced = candidate[start : end + 1]
            try:
                json.loads(sliced)
                return sliced
            except json.JSONDecodeError:
                continue
    return candidate


def _dump(value: Any) -> str:
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=str)
