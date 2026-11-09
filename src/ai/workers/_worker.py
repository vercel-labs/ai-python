"""One serializable CLI session, with direct control and the model protocol.

The CLI owns the transcript. Input and output are separate: ``send`` accepts
input, and one reader consumes ``events`` across turns. ``stream`` uses those
same primitives to expose one assistant message to the model API.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any, ClassVar, Literal, Self

import pydantic
from pydantic import TypeAdapter

from ..agents import hooks
from ..agents import runtime as runtime_
from ..types import events as events_
from ..types import messages as messages_
from ..types import tools as tools_
from ..types import usage as usage_
from . import _adapter, _bridge
from . import _errors as errors

Effort = Literal["low", "medium", "high", "max"]
BATCH_WINDOW = 0.05
OUTPUT_RETRIES = 2
OUTPUT_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON value matching this schema, "
    "no prose, no code fences:\n"
)


class UserMessageAccepted(events_.BaseEvent):
    """The transport accepted input. This is not a model response event.

    CLI protocols differ: Claude Code confirms the input write, while Codex
    and Pi acknowledge an RPC. ``sync`` reads the authoritative stored text.
    """

    kind: Literal["user_message_accepted"] = "user_message_accepted"
    message: messages_.Message


SessionEvent = events_.Event | events_.HookEvent | UserMessageAccepted


@dataclasses.dataclass
class _Call:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    answer: asyncio.Future[tuple[str, bool]]


@dataclasses.dataclass
class _Failed:
    error: BaseException


@dataclasses.dataclass
class _Accepted:
    messages: list[messages_.Message]


@dataclasses.dataclass
class _Plan:
    seed: list[messages_.Message] = dataclasses.field(default_factory=list)
    instructions: str | None = None
    results: list[messages_.ToolResultPart] = dataclasses.field(
        default_factory=list
    )
    system: list[str] = dataclasses.field(default_factory=list)
    prompt: _adapter.Prompt = dataclasses.field(default_factory=list)
    transcript: list[messages_.Message] = dataclasses.field(
        default_factory=list
    )


class _HookRuntime(runtime_.Runtime):
    """Route hooks into the same session feed as CLI output."""

    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        super().__init__()
        self.queue = queue

    async def put_event(self, event: events_.AgentEvent) -> None:
        self.queue.put_nowait(event)


@dataclasses.dataclass
class _Runtime:
    bridge: _bridge.Bridge
    queue: asyncio.Queue[Any] = dataclasses.field(default_factory=asyncio.Queue)
    send_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    in_session: bool = False
    closed: bool = False
    reader: bool = False
    stream_owner: asyncio.Task[Any] | None = None
    model_ready: asyncio.Event = dataclasses.field(
        default_factory=asyncio.Event
    )
    pump: asyncio.Task[None] | None = None
    pending: dict[str, _Call] = dataclasses.field(default_factory=dict)
    hidden: set[str] = dataclasses.field(default_factory=set)
    submission: list[messages_.Message] | None = None
    registry: hooks.HookRegistry = dataclasses.field(
        default_factory=hooks.HookRegistry
    )
    approvals: set[asyncio.Task[Any]] = dataclasses.field(default_factory=set)


class Worker(pydantic.BaseModel):
    """Base of ClaudeCode, Codex and Pi. One object represents one session.

    Dump and validate to resume a session. Use ``await worker.fork()`` to
    create an independent native session; copying is not a fork.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    worker_class_id: str
    model_name: str | None = None  # None selects the CLI default.
    cwd: str = pydantic.Field(default_factory=os.getcwd)
    env: dict[str, str] = pydantic.Field(default_factory=dict)
    executable: str | None = None  # A PATH command or a filesystem path.
    effort: Effort | None = None
    mcp_servers: dict[str, dict[str, Any]] = pydantic.Field(
        default_factory=dict
    )
    approve: bool = False  # ToolApproval hooks, also available outside Agent.
    session_id: str | None = None
    messages: list[messages_.Message] = pydantic.Field(default_factory=list)

    _classes: ClassVar[dict[str, type[Worker]]] = {}
    _runtime: _Runtime | None = pydantic.PrivateAttr(default=None)
    _cli_version: str | None = pydantic.PrivateAttr(default=None)
    _open_lock: asyncio.Lock = pydantic.PrivateAttr(
        default_factory=asyncio.Lock
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201
        super().__pydantic_init_subclass__(**kwargs)
        name = cls.model_fields["worker_class_id"].default
        if isinstance(name, str):
            Worker._classes[name] = cls

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _of_class(
        cls, data: Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Any:
        if cls is Worker and isinstance(data, dict):
            worker_class = Worker._classes.get(str(data.get("worker_class_id")))
            if worker_class is None:
                raise ValueError(
                    f"unknown worker class {data.get('worker_class_id')!r}"
                )
            return worker_class.model_validate(data)
        return handler(data)

    def __copy__(self) -> Self:
        # Revalidation resets ALL private transport state, not just the base.
        return type(self).model_validate(self.model_dump())

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self.__copy__()

    # Private transport operations. Concrete workers own their CLI processes.
    async def _cli_start(self, mcp_servers: dict[str, dict[str, Any]]) -> None:
        raise errors.UnsupportedError("this worker cannot start a CLI")

    async def _cli_close(self) -> None:
        pass

    async def _cli_open(
        self,
        *,
        session_id: str | None,
        fork: bool,
        seed: list[messages_.Message] | None,
        instructions: str | None,
    ) -> str:
        raise errors.UnsupportedError("this worker cannot open a session")

    def _cli_turn(
        self, prompt: _adapter.Prompt
    ) -> AsyncIterator[events_.Event]:
        raise errors.UnsupportedError("this worker cannot send input")

    async def _cli_send_running(self, prompt: _adapter.Prompt) -> bool:
        raise errors.UnsupportedError("this worker cannot send during a turn")

    async def _cli_interrupt(self) -> None:
        raise errors.UnsupportedError("this worker cannot interrupt")

    async def _cli_history(self, session_id: str) -> list[messages_.Message]:
        raise errors.UnsupportedError("this worker cannot read history")

    async def _cli_list_sessions(self) -> list[_adapter.SessionInfo]:
        raise errors.UnsupportedError("this worker cannot list sessions")

    def _cli_tui_command(
        self,
        session_id: str | None,
    ) -> tuple[list[str], dict[str, str], str | None]:
        raise errors.UnsupportedError("this worker has no TUI command")

    @property
    def id(self) -> str:
        return self.model_name or "default"

    @property
    def provider_name(self) -> str:
        return self.worker_class_id

    @property
    def version(self) -> str | None:
        return self._cli_version

    @property
    def hook_registry(self) -> hooks.HookRegistry:
        """Pass to ``ai.resolve_hook(..., registry=worker.hook_registry)``."""
        return self._require_runtime().registry

    async def open(self) -> None:
        """Start the CLI transport once. A session opens on first input."""
        async with self._open_lock:
            if self._runtime is not None:
                return
            bridge = _bridge.Bridge(self._on_call)
            await bridge.start()
            rt = _Runtime(bridge=bridge)
            self._runtime = rt
            try:
                await self._cli_start(
                    {
                        **self.mcp_servers,
                        _bridge.SERVER_NAME: {
                            "type": "http",
                            "url": bridge.url,
                        },
                    }
                )
            except BaseException:
                self._runtime = None
                try:
                    await self._cli_close()
                finally:
                    await bridge.close()
                raise

    async def close(self) -> None:
        """Stop the CLI and unblock the reader. Keep serializable state."""
        async with self._open_lock:
            rt = self._runtime
            if rt is None:
                return
            await self._abandon()
            rt.closed = True
            rt.queue.put_nowait(None)
            try:
                await self._cli_close()
            finally:
                self._runtime = None
                await rt.bridge.close()

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _require_runtime(self) -> _Runtime:
        if self._runtime is None:
            raise errors.WorkerError(
                f"{self.worker_class_id} worker is not open"
            )
        return self._runtime

    async def _ensure_session(self, plan: _Plan | None = None) -> None:
        rt = self._require_runtime()
        if rt.in_session:
            return
        plan = plan or _Plan()
        self.session_id = await self._cli_open(
            session_id=self.session_id,
            fork=False,
            seed=plan.seed if self.session_id is None and plan.seed else None,
            instructions=plan.instructions,
        )
        rt.in_session = True

    async def send(self, prompt: str | _adapter.Prompt) -> None:
        """Submit input and wait for transport acceptance, not the answer.

        While a turn runs this is guidance; when idle it starts a turn.
        Read ``events`` to observe acceptance and update ``messages``.
        """
        parts: _adapter.Prompt = (
            [messages_.TextPart(text=prompt)]
            if isinstance(prompt, str)
            else list(prompt)
        )
        if not parts:
            raise ValueError("send(): prompt must not be empty")
        await self.open()
        rt = self._require_runtime()
        if (
            rt.stream_owner is not None
            and rt.stream_owner is not asyncio.current_task()
        ):
            # Initial model input must open the session before another task
            # can steer it. Otherwise both paths can create a native session.
            await rt.model_ready.wait()
        async with rt.send_lock:
            accepted = rt.submission
            rt.submission = None
            if accepted is None:
                accepted = [messages_.Message(role="user", parts=list(parts))]
            await self._ensure_session()
            if rt.pump is not None and not rt.pump.done():
                if await self._cli_send_running(parts):
                    rt.queue.put_nowait(_Accepted(accepted))
                    return
                # The CLI confirmed idle without writing. Let the old reader
                # finish its terminal event before starting another turn.
                await rt.pump
            ready: asyncio.Future[None] = (
                asyncio.get_running_loop().create_future()
            )

            async def pump() -> None:
                try:
                    async for event in self._cli_turn(parts):
                        if (
                            isinstance(event, events_.StreamStart)
                            and not ready.done()
                        ):
                            rt.queue.put_nowait(_Accepted(accepted))
                            ready.set_result(None)
                        rt.queue.put_nowait(event)
                    if not ready.done():
                        raise errors.WorkerError(
                            "CLI ended before accepting input"
                        )
                except asyncio.CancelledError:
                    if not ready.done():
                        ready.cancel()
                    raise
                except BaseException as exc:
                    if not ready.done():
                        ready.set_exception(exc)
                    else:
                        rt.queue.put_nowait(_Failed(exc))

            rt.pump = asyncio.create_task(pump())
            try:
                await ready
            except asyncio.CancelledError:
                await self._abandon()
                raise

    async def events(self) -> AsyncGenerator[SessionEvent]:
        """Read the session feed until close; exactly one reader is allowed.

        StreamStart/StreamEnd mark assistant messages. UserMessageAccepted
        records input. HookEvent carries approvals. Closing/cancelling a
        reader interrupts a running turn, except when Agent tools must run.
        """
        await self.open()
        rt = self._require_runtime()
        if rt.reader or (
            rt.stream_owner is not None
            and rt.stream_owner is not asyncio.current_task()
        ):
            raise errors.WorkerError(
                "events(): this session already has a reader"
            )
        rt.reader = True
        hydrator: events_.MessageHydrator | None = None

        def out(event: events_.Event) -> events_.Event:
            nonlocal hydrator
            if hydrator is None:
                hydrator = events_.MessageHydrator()
            event = event.model_copy(update={"message": hydrator.message})
            hydrator.feed(event)
            if all(m.id != hydrator.message.id for m in self.messages):
                self.messages.append(hydrator.message)
            return event

        try:
            while True:
                item = await rt.queue.get()
                if item is None:
                    return
                if isinstance(item, _Failed):
                    raise item.error
                if isinstance(item, _Accepted):
                    known = {m.id for m in self.messages}
                    for message in item.messages:
                        if message.id not in known:
                            # Guidance belongs before the in-progress answer,
                            # not after the final answer in the model history.
                            if (
                                hydrator is not None
                                and hydrator.message in self.messages
                            ):
                                self.messages.insert(
                                    self.messages.index(hydrator.message),
                                    message,
                                )
                            else:
                                self.messages.append(message)
                            known.add(message.id)
                            if message.role == "user":
                                yield UserMessageAccepted(message=message)
                    continue
                if isinstance(item, events_.HookEvent):
                    yield item
                    continue
                if isinstance(item, events_.StreamStart):
                    hydrator = None
                    yield out(item)
                    continue
                if hydrator is None:
                    yield out(events_.StreamStart())
                if isinstance(item, _Call):
                    calls = [item]
                    with contextlib.suppress(TimeoutError):
                        async with asyncio.timeout(BATCH_WINDOW):
                            while True:
                                more = await rt.queue.get()
                                if isinstance(more, _Call):
                                    calls.append(more)
                                else:
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
                    yield out(
                        events_.StreamEnd(
                            finish_reason="tool_call",
                            provider_metadata=self._metadata(None),
                        )
                    )
                    hydrator = None
                    continue
                if isinstance(item, events_.StreamEnd):
                    finish = item.finish_reason
                    rt.hidden.clear()
                    yield out(
                        item.model_copy(
                            update={
                                "finish_reason": "other"
                                if finish == "cancelled"
                                else finish,
                                "provider_metadata": {
                                    **(item.provider_metadata or {}),
                                    **self._metadata(finish),
                                },
                            }
                        )
                    )
                    hydrator = None
                    continue
                for event in self._visible(item):
                    yield out(event)
        finally:
            rt.reader = False
            if not rt.closed and not rt.pending:
                await self._abandon()

    async def interrupt(self) -> None:
        """Request a clean turn end. The current reader receives StreamEnd."""
        rt = self._runtime
        if rt is None or rt.pump is None or rt.pump.done():
            return
        for task in list(rt.approvals):
            task.cancel()
        self._answer([])
        await self._cli_interrupt()

    async def _abandon(self) -> None:
        rt = self._runtime
        if rt is None:
            return
        for task in list(rt.approvals):
            task.cancel()
        self._answer([])
        pump = rt.pump
        if pump is not None and not pump.done():
            with contextlib.suppress(Exception):
                await self._cli_interrupt()
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
        rt.pump = None
        # A cancelled reader must not hand its queued terminal event to the
        # next model request. The partial mirror remains until sync().
        while not rt.queue.empty():
            item = rt.queue.get_nowait()
            if isinstance(item, _Call) and not item.answer.done():
                item.answer.set_result(("cancelled by the caller", True))
        rt.hidden.clear()

    async def sync(self) -> None:
        """Replace the mirror from the CLI store, for example after TUI use."""
        await self.open()
        rt = self._require_runtime()
        if rt.reader or (rt.pump is not None and not rt.pump.done()):
            raise errors.WorkerError("sync(): stop the turn and reader first")
        if self.session_id is None:
            raise errors.WorkerError("sync(): no session_id to read")
        self.messages = await self._cli_history(self.session_id)
        # Native history supersedes output queued before synchronization.
        await self._abandon()

    async def fork(self) -> Self:
        """Create a native session with a new id. Never fork a running turn."""
        if self.session_id is None:
            raise errors.WorkerError("fork(): no session_id to fork")
        rt = self._runtime
        if rt is not None and (
            rt.reader or (rt.pump is not None and not rt.pump.done())
        ):
            raise errors.WorkerError("fork(): stop the turn and reader first")
        branch = self.__copy__()
        try:
            await branch.open()
            branch.session_id = await branch._cli_open(
                session_id=self.session_id,
                fork=True,
                seed=None,
                instructions=None,
            )
            if branch.session_id == self.session_id:
                raise errors.WorkerError(
                    "fork(): CLI did not create a new session id"
                )
            branch._require_runtime().in_session = True
            branch.messages = await branch._cli_history(branch.session_id)
            return branch
        except BaseException:
            await branch.close()
            raise

    def tui_command(self) -> tuple[list[str], dict[str, str]]:
        """Return argv and env. Close this worker before starting the TUI."""
        argv, env, minted = self._cli_tui_command(self.session_id)
        if self.session_id is None:
            if minted is None:
                raise errors.WorkerError(
                    "CLI chooses its session id: send input first"
                )
            self.session_id = minted
        return argv, env

    async def tui(self) -> int:
        await self.close()
        argv, env = self.tui_command()
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.cwd, env={**os.environ, **env}
        )
        code = await proc.wait()
        await self.sync()
        return code

    @classmethod
    async def list_sessions(
        cls, cwd: str, **fields: Any
    ) -> list[_adapter.SessionInfo]:
        worker = cls(cwd=cwd, **fields)
        async with worker:
            return await worker._cli_list_sessions()

    async def stream(
        self,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
    ) -> AsyncGenerator[events_.Event]:
        """Send only new input and read one assistant message via events()."""
        await self.open()
        rt = self._require_runtime()
        if rt.reader or rt.stream_owner is not None:
            raise errors.WorkerError(
                "stream(): this session already has a reader"
            )
        if not rt.queue.empty() and not rt.pending:
            raise errors.WorkerError(
                "stream(): unread session events; "
                "consume events() or sync() first"
            )
        rt.stream_owner = asyncio.current_task()
        rt.model_ready.clear()
        # Transport reader tasks can predate Agent.run(). Use the current
        # consumer's registry, not the callback task's inherited context.
        with contextlib.suppress(LookupError):
            rt.registry = hooks.get_hook_registry()
        try:
            plan = self._plan(messages)
            rt.bridge.set_tools(
                [t for t in tools or () if t.kind == "function"]
            )
            if rt.pending:
                self._answer(plan.results)
                for message in plan.transcript:
                    if all(m.id != message.id for m in self.messages):
                        self.messages.append(message)
            if plan.prompt:
                if rt.in_session and plan.system:
                    plan.prompt.insert(
                        0,
                        messages_.TextPart(
                            text="\n\n".join(plan.system) + "\n\n"
                        ),
                    )
                await self._ensure_session(plan)
                rt.submission = plan.transcript
                prompt = list(plan.prompt)
                if output_type is not None:
                    prompt.append(
                        messages_.TextPart(
                            text=OUTPUT_INSTRUCTION
                            + json.dumps(TypeAdapter(output_type).json_schema())
                        )
                    )
                await self.send(prompt)
            elif not plan.results:
                raise ValueError(
                    "stream(): no new user message or tool result to send"
                )
            rt.model_ready.set()

            answer: list[str] = []
            # events() owns the native mirror; model output must hydrate a
            # separate message, retaining the native id even across retries.
            projection = events_.MessageHydrator()
            async with contextlib.aclosing(self.events()) as feed:
                async for event in feed:
                    if isinstance(event, UserMessageAccepted):
                        continue
                    if isinstance(event, events_.HookEvent):
                        # Agent's outer runtime carries hook events, while the
                        # direct path exposes them in events().
                        try:
                            runtime = runtime_.get_runtime()
                        except LookupError:
                            raise errors.WorkerError(
                                "approval requires events() or ai.Agent.run(); "
                                "resolve with worker.hook_registry"
                            ) from None
                        await runtime.put_event(event)
                        continue
                    if isinstance(event, events_.StreamEnd):
                        if (
                            output_type is not None
                            and event.finish_reason != "tool_call"
                        ):
                            text, usage = await self._output(
                                "".join(answer), output_type, event.usage, feed
                            )
                            yield projection.feed(
                                events_.TextStart(
                                    block_id="output", message=event.message
                                )
                            )
                            yield projection.feed(
                                events_.TextDelta(
                                    chunk=text,
                                    block_id="output",
                                    message=event.message,
                                )
                            )
                            yield projection.feed(
                                events_.TextEnd(
                                    block_id="output", message=event.message
                                )
                            )
                            event = event.model_copy(update={"usage": usage})
                        native = event.message
                        event = projection.feed(event)
                        if output_type is not None:
                            # Persist all emitted parts, including builtin and
                            # Agent tools, not just the validated output text.
                            metadata = native.provider_metadata or {}
                            native.provider_metadata = {
                                **metadata,
                                "worker": {
                                    **metadata.get("worker", {}),
                                    "model_parts": event.message.model_dump(
                                        mode="json"
                                    )["parts"],
                                },
                            }
                        yield event
                        break
                    if output_type is not None and isinstance(
                        event,
                        events_.TextStart | events_.TextDelta | events_.TextEnd,
                    ):
                        if isinstance(event, events_.TextDelta):
                            answer.append(event.chunk)
                        continue
                    if isinstance(event, events_.BuiltinToolStart):
                        answer.clear()
                    yield projection.feed(event)
        finally:
            rt.stream_owner = None
            rt.submission = None
            rt.model_ready.set()

    def _plan(self, messages: list[messages_.Message]) -> _Plan:
        known = [m.id for m in self.messages]
        given = [m.id for m in messages]
        plan = _Plan()
        plan.instructions = (
            "\n\n".join(m.text for m in messages if m.role == "system") or None
        )
        if known:
            projected: list[messages_.Message] = []
            for message in self.messages:
                metadata = (message.provider_metadata or {}).get("worker", {})
                if metadata.get("model_hidden"):
                    continue
                parts = metadata.get("model_parts")
                projected.append(
                    message
                    if parts is None
                    else message.model_copy(
                        update={
                            "parts": TypeAdapter(
                                list[messages_.Part]
                            ).validate_python(parts)
                        }
                    )
                )
            matched_ids = False
            for history in (self.messages, projected):
                if given[: len(history)] != [m.id for m in history]:
                    continue
                matched_ids = True
                # Accept either complete view, never arbitrary per-message
                # substitutions. Matching ids must not conceal changed parts.
                if all(
                    old.role == new.role and old.parts == new.parts
                    for old, new in zip(history, messages, strict=False)
                ):
                    new_messages = messages[len(history) :]
                    break
            else:
                raise errors.TranscriptMismatchError(
                    "stored message content was changed"
                    if matched_ids
                    else f"{self.worker_class_id}: messages do not continue "
                    "this session; use a new worker"
                )
        elif self.session_id is not None:
            new_messages = list(messages)
        else:
            cut = len(messages)
            while cut and messages[cut - 1].role in ("user", "system", "tool"):
                cut -= 1
            plan.seed = [m for m in messages[:cut] if m.role != "system"]
            new_messages = messages[cut:]
            plan.transcript = list(messages[:cut])
        for message in new_messages:
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
                    "only user, system and tool messages may follow "
                    "the last answer"
                )
        plan.transcript.extend(new_messages)
        return plan

    def _answer(self, results: list[messages_.ToolResultPart]) -> None:
        rt = self._require_runtime()
        for part in results:
            call = rt.pending.pop(part.tool_call_id, None)
            if call is None or call.answer.done():
                continue
            value = part.model_input
            if value is messages_.MODEL_INPUT_UNSET:
                value = part.result
            call.answer.set_result(
                (
                    value if isinstance(value, str) else _dump(value),
                    part.result_kind == "error",
                )
            )
        for call in rt.pending.values():
            if not call.answer.done():
                call.answer.set_result(("no result was given", True))
        rt.pending.clear()

    async def _on_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        rt = self._runtime
        if rt is None or rt.pump is None:
            return "no turn is running", True
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
        if self._is_bridge_tool(tool_name):
            return True, ""
        rt = self._runtime
        if rt is None or rt.closed:
            return False, "session is closed"
        task = asyncio.current_task()
        if task is not None:
            rt.approvals.add(task)
        try:
            approval = await hooks.hook(
                f"approve_{tool_call_id}",
                payload=hooks.ToolApproval,
                metadata={"tool": tool_name, "kwargs": tool_input},
                tool_call_id=tool_call_id,
                registry=rt.registry,
                runtime=_HookRuntime(rt.queue),
            )
            return approval.granted, approval.reason or "denied"
        except asyncio.CancelledError:
            return False, "the approval was cancelled"
        except Exception as exc:
            return False, f"approval failed ({type(exc).__name__}); denied"
        finally:
            if task is not None:
                rt.approvals.discard(task)

    def _is_bridge_tool(self, tool_name: str) -> bool:
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
        rt = self._require_runtime()
        match event:
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
        feed: AsyncIterator[SessionEvent],
    ) -> tuple[str, usage_.Usage | None]:
        parsed, error = _parse(text, output_type)
        attempt = 0
        while error is not None and attempt < OUTPUT_RETRIES:
            attempt += 1
            await self.send(
                f"Your previous reply failed validation: {error}\n"
                "Reply again with ONLY the corrected JSON."
            )
            chunks: list[str] = []
            async for event in feed:
                if isinstance(event, UserMessageAccepted | events_.ModelEvent):
                    # Retry input and answers remain in the native mirror, but
                    # are not separate messages in the caller's model view.
                    # Reapply on StreamEnd, which installs native metadata.
                    metadata = event.message.provider_metadata or {}
                    event.message.provider_metadata = {
                        **metadata,
                        "worker": {
                            **metadata.get("worker", {}),
                            "model_hidden": True,
                        },
                    }
                if isinstance(event, events_.TextDelta):
                    chunks.append(event.chunk)
                elif isinstance(event, events_.BuiltinToolStart):
                    chunks.clear()
                elif isinstance(event, events_.StreamEnd):
                    if event.usage:
                        usage = usage + event.usage if usage else event.usage
                    break
                elif isinstance(event, events_.HookEvent):
                    await runtime_.get_runtime().put_event(event)
            text = "".join(chunks)
            parsed, error = _parse(text, output_type)
        return (parsed.model_dump_json() if parsed else text), usage

    def _metadata(self, finish: str | None) -> dict[str, Any]:
        meta: dict[str, Any] = {"worker_class_id": self.worker_class_id}
        if self.session_id:
            meta["session_id"] = self.session_id
        if finish is not None:
            meta["finish_reason"] = finish
        return {"worker": meta}


def _parse(
    text: str, output_type: type[pydantic.BaseModel]
) -> tuple[pydantic.BaseModel | None, Exception | None]:
    try:
        return output_type.model_validate_json(_extract_json(text)), None
    except Exception as exc:
        return None, exc


def _extract_json(text: str) -> str:
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
