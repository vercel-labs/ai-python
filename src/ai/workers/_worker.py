"""`Worker`: one session of a coding-agent CLI.

Two public layers live here (see DESIGN.md):

- L1, direct control: `start`, `send`, `events`, `interrupt`, `close`, and
  the state `session_id` and `messages`.
- L2, the model protocol: `stream`, built one time on L1.

A concrete worker (`Codex`, ...) only adds settings and a private wire
(`_wire.Wire`) that speaks its CLI's protocol.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import os
import warnings
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Sequence,
)
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import pydantic

from .. import experimental_telemetry as telemetry
from ..types import builders
from ..types import events as events_
from ..types import messages as messages_
from ..types import tools as tools_
from ..types import usage as usage_
from . import _bridge, _handoff, _wire
from . import _errors as errors

if TYPE_CHECKING:
    from ..models.core import params as params_

# Seconds to wait for more `ai` tool calls before the message ends: a CLI
# that calls tools in parallel sends the calls one by one.
BATCH_WINDOW = 0.05
# Seconds `interrupt()` waits for the CLI to end the turn before it closes it.
INTERRUPT_TIMEOUT = 30.0
# Key in `provider_metadata` for what the worker records about a message.
METADATA_KEY = "worker"

NEW_MESSAGE = "No result: a new user message came before the result."
INTERRUPTED = "No result: the turn was interrupted."
CLOSED = "No result: the worker closed."


class MessageEvent(events_.BaseEvent):
    """A complete message went into `worker.messages`."""

    message: messages_.Message
    kind: Literal["worker_message"] = "worker_message"


WorkerEvent = events_.Event | MessageEvent


class WorkerSessionSpanData(pydantic.BaseModel):
    """One worker session, from `start()` to `close()`."""

    kind: Literal["worker_session"] = "worker_session"
    worker: str
    model: str | None = None
    cwd: str
    session_id: str | None = None


class WorkerMessageSpanData(pydantic.BaseModel):
    """One assistant message of a worker."""

    kind: Literal["worker_message"] = "worker_message"
    worker: str
    model: str | None = None
    session_id: str | None = None
    message: messages_.Message | None = None
    usage: usage_.Usage | None = None
    finish_reason: str | None = None


def is_interrupted(message: messages_.Message) -> bool:
    """Tell if an interrupt ended `message`, or the worker made it up.

    Such messages are in `worker.messages`, but `ai.Agent` does not keep
    them. `stream()` ignores them when it compares histories.
    """
    data = (message.provider_metadata or {}).get(METADATA_KEY) or {}
    return bool(data.get("interrupted"))


@dataclasses.dataclass
class _Call:
    """An `ai` tool call that the CLI made through the bridge. The CLI waits."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    answer: asyncio.Future[tuple[str, bool]]


@dataclasses.dataclass
class _Command:
    """Work for the pump: only the pump task changes the conversation."""

    run: Callable[[], Awaitable[None]]
    done: asyncio.Future[None]


@dataclasses.dataclass(frozen=True)
class _Stopped:
    """The wire stopped: the CLI exited or failed."""

    error: BaseException


_END = object()  # put on a reader's queue at close


class _Session:
    """A started worker: the CLI wire, the bridge, the pump and the readers.

    The pump is the only task that changes the conversation state. Other
    tasks give it items through `queue`: wire items, bridge calls, commands.
    """

    def __init__(self, worker: Worker, wire: _wire.Wire) -> None:
        self.worker = worker
        self.wire = wire
        self.bridge = _bridge.Bridge(self._on_call)
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.readers: set[asyncio.Queue[Any]] = set()
        self.tasks: list[asyncio.Task[None]] = []
        self.span: telemetry.Span[WorkerSessionSpanData] = (
            telemetry.create_span(
                WorkerSessionSpanData(
                    worker=worker.worker_class_id,
                    model=worker.model_name,
                    cwd=worker.cwd,
                    session_id=worker.session_id,
                ),
                set_as_current=False,
            )
        )
        # Set when no turn runs. `interrupt()` waits on it.
        self.idle = asyncio.Event()
        self.idle.set()
        self.interrupting = False
        self.stopped: BaseException | None = None
        # A new CLI session, not one that continues `session_id`.
        self.fresh = worker.session_id is None
        # The tools the CLI saw when it connected to the bridge.
        self.served: list[dict[str, Any]] = []
        # True after the first user message went to the CLI.
        self.used = False
        # Turn errors, by the id of the assistant message they ended.
        self.failures: dict[str, BaseException] = {}

        # The open assistant message.
        self.hydrator: events_.MessageHydrator | None = None
        self.message_id = ""
        self.message_span: telemetry.Span[WorkerMessageSpanData] | None = None
        self.tool_spans: dict[
            str, telemetry.Span[telemetry.ToolExecutionSpanData]
        ] = {}
        # Text and reasoning blocks and CLI tool calls that did not end yet.
        # A steer waits for none to be open, so no block is cut in two.
        self.open: set[str] = set()
        # User messages that the CLI took while the message was busy.
        self.inputs: list[messages_.Message] = []
        # `ai` tool calls without a result. While there are some, other items
        # wait in `held`: they come after the results.
        self.calls: dict[str, _Call] = {}
        self.held: list[Any] = []
        # When the message with `ai` tool calls ends, if no call comes first.
        self.close_at: float | None = None

    # -- life -----------------------------------------------------------------

    async def start(self) -> None:
        worker = self.worker
        self.span.stamp_start()
        await self.span.push()
        await self.bridge.start()
        self.bridge.set_tools(worker.tools)
        self.served = _tool_specs(worker.tools)
        system = [m.text for m in worker.messages if m.role == "system"]
        instructions = "\n\n".join(
            t for t in (worker.instructions, *system) if t
        )
        seed = (
            [m for m in worker.messages if m.role != "system"]
            if worker.session_id is None
            else []
        )
        try:
            worker.session_id = await self.wire.start(
                bridge_url=self.bridge.url,
                instructions=instructions or None,
                seed=seed,
            )
        except BaseException as exc:
            with contextlib.suppress(Exception):
                await self.wire.close()
            with contextlib.suppress(Exception):
                await self.bridge.close()
            await self.span.stamp_end(error=exc).push()
            raise
        self.span.data.session_id = worker.session_id
        self.tasks = [
            asyncio.create_task(self._read()),
            asyncio.create_task(self._pump()),
        ]

    async def close(self) -> None:
        if self.stopped is None:
            # End what is open, so the state stays valid.
            with contextlib.suppress(Exception):
                await self.do(self._shut)
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(BaseException):
                await task
        with contextlib.suppress(Exception):
            await self.wire.close()
        with contextlib.suppress(Exception):
            await self.bridge.close()
        self.stopped = self.stopped or errors.WorkerError(
            "the worker is closed"
        )
        self.idle.set()
        self._fail_commands()
        for reader in self.readers:
            reader.put_nowait(_END)
        await self.span.stamp_end().push()

    async def _shut(self) -> None:
        await self.fail_calls(CLOSED)
        if self.hydrator is not None:
            await self._end_message("other", interrupted=True)

    async def do(self, run: Callable[[], Awaitable[None]]) -> None:
        """Run `run` in the pump task and wait for it."""
        if self.stopped is not None:
            raise errors.WorkerError(f"the worker stopped: {self.stopped}")
        done = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(_Command(run, done))
        await done

    def _fail_commands(self) -> None:
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if isinstance(item, _Command) and not item.done.done():
                item.done.set_exception(
                    errors.WorkerError(f"the worker stopped: {self.stopped}")
                )

    def _broadcast(self, item: Any) -> None:
        for reader in self.readers:
            reader.put_nowait(item)

    # -- tasks ----------------------------------------------------------------

    async def _read(self) -> None:
        error: BaseException
        try:
            async for item in self.wire.events():
                self.queue.put_nowait(item)
            error = errors.WorkerError(
                f"{self.worker.worker_class_id}: the CLI stopped"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        self.queue.put_nowait(_Stopped(error))

    async def _pump(self) -> None:
        while True:
            try:
                async with asyncio.timeout_at(self.close_at):
                    item = await self.queue.get()
            except TimeoutError:
                self.close_at = None
                if self.hydrator is not None and self.calls:
                    await self._end_message("tool_call")
                continue
            if isinstance(item, _Command):
                try:
                    await item.run()
                except Exception as exc:
                    if not item.done.done():
                        item.done.set_exception(exc)
                else:
                    if not item.done.done():
                        item.done.set_result(None)
                continue
            try:
                if isinstance(item, _Stopped):
                    raise item.error
                await self._handle(item)
            except Exception as exc:
                await self._stop(exc)
                return

    async def _stop(self, error: BaseException) -> None:
        if not isinstance(error, errors.WorkerError):
            wrapped = errors.WorkerError(
                f"{self.worker.worker_class_id} failed: {error}"
            )
            wrapped.__cause__ = error
            error = wrapped
        for call in self.calls.values():
            if not call.answer.done():
                call.answer.set_result((str(error), True))
        self.calls.clear()
        if self.hydrator is not None:
            with contextlib.suppress(Exception):
                await self._end_message("error", error=error)
        self.stopped = error
        self.idle.set()
        self._fail_commands()
        for reader in self.readers:
            reader.put_nowait(error)

    async def _on_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        """Report a bridge call to an `ai` tool; wait for its result."""
        if self.stopped is not None:
            return f"the worker stopped: {self.stopped}", True
        call = _Call(
            tool_call_id=messages_.generate_id("call"),
            name=name,
            arguments=arguments,
            answer=asyncio.get_running_loop().create_future(),
        )
        self.queue.put_nowait(call)
        return await call.answer

    # -- the conversation (pump task only) ------------------------------------

    async def _handle(self, item: Any) -> None:
        if self.calls and (
            self.hydrator is None or not isinstance(item, _Call)
        ):
            # Waiting for `ai` tool results: what comes now goes after them.
            if self.hydrator is not None:
                await self._end_message("tool_call")
            self.held.append(item)
            return
        match item:
            case _Call():
                await self._call(item)
            case _wire.Accepted(message=message):
                await self._accept(message)
            case _wire.TurnEnd():
                await self._turn_end(item)
            case _:
                await self._content(item)

    async def _content(self, event: _wire.Content) -> None:
        if self.hydrator is None:
            await self._begin_message()
        self.idle.clear()
        match event:
            case (
                events_.TextStart(block_id=block)
                | events_.ReasoningStart(block_id=block)
            ):
                self.open.add(block)
            case (
                events_.TextEnd(block_id=block)
                | events_.ReasoningEnd(block_id=block)
            ):
                self.open.discard(block)
            case events_.BuiltinToolStart(tool_call_id=call_id, tool_name=name):
                self.open.add(call_id)
                tool_span = telemetry.create_span(
                    telemetry.ToolExecutionSpanData(
                        tool_name=name, tool_call_id=call_id
                    ),
                    parent=self.message_span,
                    set_as_current=False,
                )
                self.tool_spans[call_id] = tool_span
                await tool_span.stamp_start().push()
            case events_.BuiltinToolEnd(tool_call=part):
                started = self.tool_spans.get(part.tool_call_id)
                if started is not None:
                    with contextlib.suppress(ValueError):
                        args = json.loads(part.tool_args or "{}")
                        started.data.args = (
                            args if isinstance(args, dict) else None
                        )
            case events_.BuiltinToolResult(result=result):
                self.open.discard(result.tool_call_id)
                ended = self.tool_spans.pop(result.tool_call_id, None)
                if ended is not None:
                    ended.data.result = result.result
                    ended.data.is_error = result.is_error
                    await ended.stamp_end().push()
            case _:
                pass
        self._emit(event)
        if self.inputs and not self.open:
            await self._end_message("other")

    async def _accept(self, message: messages_.Message) -> None:
        self.idle.clear()
        if self.hydrator is None:
            await self._append(message)
            return
        # A steer: the assistant message ends, the user message follows,
        # and a new assistant message starts.
        self.inputs.append(message)
        if not self.open:
            await self._end_message("other")

    async def _call(self, call: _Call) -> None:
        if self.hydrator is None:
            await self._begin_message()
        self.calls[call.tool_call_id] = call
        args = json.dumps(call.arguments)
        part = messages_.ToolCallPart(
            tool_call_id=call.tool_call_id, tool_name=call.name, tool_args=args
        )
        self._emit(
            events_.ToolStart(
                tool_call_id=call.tool_call_id, tool_name=call.name
            )
        )
        self._emit(
            events_.ToolDelta(tool_call_id=call.tool_call_id, chunk=args)
        )
        self._emit(
            events_.ToolEnd(tool_call_id=call.tool_call_id, tool_call=part)
        )
        self.close_at = asyncio.get_running_loop().time() + BATCH_WINDOW

    async def _turn_end(self, end: _wire.TurnEnd) -> None:
        interrupted = end.error is None and (
            end.reason == "interrupted" or self.interrupting
        )
        reason: str = "other" if interrupted else end.reason
        # Every turn ends with one StreamEnd, also a turn that said nothing.
        await self._end_message(
            reason, usage=end.usage, interrupted=interrupted, error=end.error
        )
        self.interrupting = False
        self.idle.set()

    async def answer(self, message: messages_.Message) -> None:
        """Give `ai` tool results to the CLI calls that wait for them."""
        results = message.tool_results
        if not results:
            raise ValueError("a tool message needs at least one tool result")
        unknown = [
            p.tool_call_id for p in results if p.tool_call_id not in self.calls
        ]
        if unknown:
            raise ValueError(f"no tool call waits for these results: {unknown}")
        if self.hydrator is not None:
            await self._end_message("tool_call")
        for part in results:
            call = self.calls.pop(part.tool_call_id)
            value = part.model_input
            if value is messages_.MODEL_INPUT_UNSET:
                value = part.result
            if not call.answer.done():
                call.answer.set_result(
                    (_handoff.result_text(value), part.result_kind == "error")
                )
        await self._append(message)
        if not self.calls:
            await self._release()

    async def fail_calls(self, reason: str) -> None:
        """Answer the waiting `ai` tool calls with an error."""
        if not self.calls:
            return
        if self.hydrator is not None:
            await self._end_message("tool_call")
        parts: list[messages_.Part] = []
        for call in self.calls.values():
            if not call.answer.done():
                call.answer.set_result((reason, True))
            parts.append(
                messages_.ToolResultPart(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.name,
                    result=reason,
                    result_kind="error",
                )
            )
        self.calls.clear()
        # A tool message must follow the calls, or no model takes the history.
        await self._append(
            messages_.Message(
                role="tool",
                parts=parts,
                provider_metadata={METADATA_KEY: {"interrupted": True}},
            )
        )
        await self._release()

    async def _release(self) -> None:
        held, self.held = self.held, []
        for item in held:
            await self._handle(item)

    async def _begin_message(self) -> None:
        worker = self.worker
        self.message_id = messages_.generate_id("msg")
        self.hydrator = events_.MessageHydrator()
        self.open.clear()
        self.message_span = telemetry.create_span(
            WorkerMessageSpanData(
                worker=worker.worker_class_id,
                model=worker.model_name,
                session_id=worker.session_id,
            ),
            parent=self.span,
            set_as_current=False,
        )
        await self.message_span.stamp_start().push()
        self._emit(events_.StreamStart())

    async def _end_message(
        self,
        reason: str,
        *,
        usage: usage_.Usage | None = None,
        interrupted: bool = False,
        error: BaseException | None = None,
    ) -> None:
        if self.hydrator is None:
            await self._begin_message()
        assert self.hydrator is not None
        self.close_at = None
        metadata: dict[str, Any] | None = None
        if error is not None:
            metadata = {METADATA_KEY: {"error": str(error)}}
            self.failures[self.message_id] = error
        elif interrupted:
            metadata = {METADATA_KEY: {"interrupted": True}}
        self._emit(
            events_.StreamEnd(
                finish_reason=reason, usage=usage, provider_metadata=metadata
            )
        )
        message = self.hydrator.message
        self.hydrator = None
        self.open.clear()
        for span in self.tool_spans.values():
            await span.stamp_end().push()
        self.tool_spans.clear()
        if self.message_span is not None:
            self.message_span.data.message = message
            self.message_span.data.usage = message.usage
            self.message_span.data.finish_reason = reason
            await self.message_span.stamp_end(error=error).push()
            self.message_span = None
        await self._append(message)
        inputs, self.inputs = self.inputs, []
        if reason == "tool_call":
            # The tool results must follow the calls: steers wait for them.
            self.held[:0] = [_wire.Accepted(m) for m in inputs]
        else:
            for m in inputs:
                await self._append(m)

    def _emit(self, event: events_.Event) -> None:
        assert self.hydrator is not None
        stamp = messages_.Message(
            id=self.message_id, role="assistant", parts=[]
        )
        self._broadcast(
            self.hydrator.feed(event.model_copy(update={"message": stamp}))
        )

    async def _append(self, message: messages_.Message) -> None:
        # A new list, not append: a copied worker must not share it.
        self.worker.messages = [*self.worker.messages, message]
        self._broadcast(MessageEvent(message=message))


async def _read_events(
    session: _Session, queue: asyncio.Queue[Any]
) -> AsyncGenerator[WorkerEvent]:
    try:
        while True:
            item = await queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        session.readers.discard(queue)


class Worker(pydantic.BaseModel):
    """One session of a coding-agent CLI.

    Use it as a model: `ai.stream(worker, messages)` and
    `ai.Agent(...).run(worker, messages)`. Or control it directly with
    `start`, `send`, `events`, `interrupt` and `close`.

    The fields are plain data. `model_dump()` saves the worker, and
    `Worker.model_validate(data)` makes the correct subclass again; it
    continues the same CLI session on the next start. `model_copy()` makes a
    worker that is not started. Two started workers on one `session_id` are
    two writers on one CLI session: set `session_id=None` on the copy to
    start a new session that is seeded from `messages`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    worker_class_id: str
    model_name: str | None = None
    """The CLI's model. None: the CLI default."""
    cwd: str = pydantic.Field(default_factory=os.getcwd)
    """Where the CLI runs."""
    env: dict[str, str] = pydantic.Field(default_factory=dict)
    """Added to the CLI's environment: keys, base URLs, config dirs."""
    executable: str | None = None
    """The CLI command or path. None: the usual command on PATH."""
    instructions: str | None = None
    """Added to the CLI's system prompt when a session starts."""
    tools: list[tools_.Tool] = pydantic.Field(default_factory=list)
    """`ai` function tools that the CLI can call through the MCP bridge."""
    session_id: str | None = None
    """The CLI's id for the session. Set it to continue a session."""
    messages: list[messages_.Message] = pydantic.Field(default_factory=list)
    """The conversation, in `ai` messages. The worker keeps it current."""

    _classes: ClassVar[dict[str, type[Worker]]] = {}
    _session: _Session | None = pydantic.PrivateAttr(default=None)
    _start_lock: asyncio.Lock = pydantic.PrivateAttr(
        default_factory=asyncio.Lock
    )
    _stream_lock: asyncio.Lock = pydantic.PrivateAttr(
        default_factory=asyncio.Lock
    )

    def __init__(self, **data: Any) -> None:
        if type(self) is Worker:
            raise TypeError(
                "Worker must be subclassed, for example workers.Codex"
            )
        super().__init__(**data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201
        super().__pydantic_init_subclass__(**kwargs)
        class_id = cls.model_fields["worker_class_id"].default
        if isinstance(class_id, str):
            existing = Worker._classes.get(class_id)
            if (
                existing is not None
                and existing.__qualname__ != cls.__qualname__
            ):
                raise RuntimeError(f"duplicate worker class id: {class_id!r}")
            Worker._classes[class_id] = cls

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _load_registered_worker(
        cls, data: Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Any:
        if cls is Worker and isinstance(data, dict):
            worker_class = Worker._classes.get(str(data.get("worker_class_id")))
            if worker_class is None:
                raise ValueError(
                    f"unknown worker_class_id: {data.get('worker_class_id')!r}"
                )
            return worker_class.model_validate(data)
        return handler(data)

    def __copy__(self) -> Self:
        copied = super().__copy__()
        copied._session = None
        copied._start_lock = asyncio.Lock()
        copied._stream_lock = asyncio.Lock()
        return copied

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = super().__deepcopy__(memo)
        copied._session = None
        copied._start_lock = asyncio.Lock()
        copied._stream_lock = asyncio.Lock()
        return copied

    def _make_wire(self) -> _wire.Wire:
        """Return a new wire for this worker's CLI. Subclasses implement it."""
        raise NotImplementedError

    # -- L1: direct control ---------------------------------------------------

    async def start(self) -> None:
        """Start the CLI and open the session.

        With `session_id`, continue that session. Without it, start a new
        session, seeded with `messages` (lossy). Start a second time: nothing
        happens. After the CLI stopped, start it again.
        """
        async with self._start_lock:
            if self._session is not None and self._session.stopped is None:
                return
            if self._session is not None:
                session, self._session = self._session, None
                await session.close()
            session = _Session(self, self._make_wire())
            await session.start()
            self._session = session

    async def send(self, message: messages_.Message | str) -> None:
        """Give a message to the CLI.

        A user message (or a `str`) starts a turn, or steers the turn that
        runs. The CLI decides. A tool message gives results to `ai` tool
        calls that wait. `send()` returns when the CLI has the message; the
        message goes into `messages` when the CLI accepts it.
        """
        session = self._require_session()
        if isinstance(message, str):
            message = builders.user_message(message)
        match message.role:
            case "tool":
                await session.do(functools.partial(session.answer, message))
            case "user":
                await session.do(
                    functools.partial(session.fail_calls, NEW_MESSAGE)
                )
                session.idle.clear()
                session.used = True
                try:
                    await session.wire.send(message)
                except BaseException:
                    if session.hydrator is None and not session.calls:
                        session.idle.set()
                    raise
            case role:
                raise ValueError(
                    f"send() takes a user or tool message, not {role!r}"
                )

    def events(self) -> AsyncGenerator[WorkerEvent]:
        """Return a new subscriber to the session's events.

        The subscription starts now, not at the first `__anext__`: call it
        before `send()` to see everything that the message causes. Many
        subscribers are permitted. It ends when the worker closes, and raises
        if the CLI fails. Close it (or let it end) when you are done.
        """
        session = self._require_session()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        session.readers.add(queue)
        return _read_events(session, queue)

    async def interrupt(self) -> None:
        """Stop the current turn. The CLI keeps the session.

        Return when the CLI ended the turn. If it does not end the turn in
        `INTERRUPT_TIMEOUT` seconds, close the worker; start it again to
        continue. No turn: nothing happens.
        """
        session = self._session
        if session is None or session.stopped is not None:
            return
        await session.do(functools.partial(session.fail_calls, INTERRUPTED))
        if session.idle.is_set():
            return
        session.interrupting = True
        try:
            async with asyncio.timeout(INTERRUPT_TIMEOUT):
                await session.wire.interrupt()
                await session.idle.wait()
        except TimeoutError:
            await self.close()

    async def close(self) -> None:
        """Stop the CLI. The state stays. A second close does nothing."""
        async with self._start_lock:
            session, self._session = self._session, None
            if session is not None:
                await session.close()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _require_session(self) -> _Session:
        session = self._session
        if session is None:
            raise errors.WorkerError("the worker is not started: call start()")
        if session.stopped is not None:
            raise errors.WorkerError(f"the worker stopped: {session.stopped}")
        return session

    # -- L2: the model protocol -----------------------------------------------

    @property
    def id(self) -> str:
        return self.model_name or "default"

    @property
    def provider_name(self) -> str:
        return self.worker_class_id

    async def stream(
        self,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events_.Event]:
        """Answer `messages` with one assistant message, as a model does.

        Use `ai.stream(worker, messages)` or `ai.Agent.run(worker, ...)`.
        The message ends with `finish_reason="tool_call"` when the CLI calls
        `ai` tools from `tools=`; the CLI waits for the results, which come
        with the next call. To cancel the stream or to stop reading it
        interrupts the turn. `params` has no effect: use worker settings.
        """
        if output_type is not None:
            raise errors.UnsupportedError(
                "workers do not support output_type yet"
            )
        if self._stream_lock.locked():
            raise errors.WorkerError("a worker runs one stream() at a time")
        async with self._stream_lock:
            new, seeds = self._new_messages(messages)
            if tools is not None:
                self.tools = [t for t in tools if t.kind == "function"]
            session = self._session
            if session is not None and session.stopped is None:
                # A CLI reads the tool list once, when it connects to the
                # bridge, and takes a past only when its session starts. If
                # either changed and no turn runs, start the CLI again.
                restart = seeds or _tool_specs(self.tools) != session.served
                if restart and session.idle.is_set() and not session.calls:
                    await self.close()
                    if session.fresh and not session.used:
                        # The session is new and empty: start a new one, which
                        # also takes the past.
                        self.session_id = None
                elif restart:
                    warnings.warn(
                        "the tools changed during a turn: the CLI keeps the "
                        "tools it saw at the start until the worker restarts",
                        stacklevel=2,
                    )
            await self.start()
            session = self._require_session()
            session.bridge.set_tools(self.tools)
            feed = self.events()
            ended = False
            try:
                for message in new:
                    await self.send(message)
                started = False
                async for event in feed:
                    if isinstance(event, MessageEvent):
                        continue
                    if isinstance(event, events_.StreamStart):
                        started = True
                    if not started:
                        continue  # the end of a message that began earlier
                    if isinstance(event, events_.StreamEnd):
                        ended = True
                        error = session.failures.pop(event.message.id, None)
                        if error is not None:
                            raise error
                        yield event
                        return
                    yield event
                raise errors.WorkerError("the worker closed during stream()")
            except (asyncio.CancelledError, GeneratorExit) as exc:
                if not ended:
                    # Shielded: a second cancel must not cut the interrupt.
                    task = asyncio.ensure_future(self.interrupt())
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        if isinstance(exc, GeneratorExit):
                            raise
                    except Exception:
                        pass
                raise
            finally:
                await feed.aclose()

    def _new_messages(
        self, messages: list[messages_.Message]
    ) -> tuple[list[messages_.Message], bool]:
        """Compare `messages` with the conversation; return the new part.

        Also return True when the past must seed a session that a started
        worker opened empty.
        """
        if not self.messages:
            # First use: the past is the conversation. Without a session_id
            # it seeds the new CLI session; with one, the CLI has it already.
            start = len(messages)
            while start and messages[start - 1].role == "user":
                start -= 1
            if start == len(messages):
                raise errors.TranscriptMismatchError(
                    "the last message must be a user message: a worker cannot "
                    "continue tool calls that it did not make"
                )
            self.messages = list(messages[:start])
            seeds = (
                bool(start)
                and self._session is not None
                and self._session.fresh
            )
            return list(messages[start:]), seeds
        skipped = {m.id for m in self.messages if is_interrupted(m)}
        known = [m.id for m in self.messages if m.id not in skipped]
        given = [m for m in messages if m.id not in skipped]
        if [m.id for m in given[: len(known)]] != known:
            raise errors.TranscriptMismatchError(
                "the messages do not continue this worker's conversation; "
                "use worker.messages as the history, or a new worker"
            )
        new = given[len(known) :]
        if not new:
            raise ValueError("stream(): no new message to send")
        for message in new:
            if message.role not in ("user", "tool"):
                raise errors.TranscriptMismatchError(
                    f"a new {message.role} message cannot go to the CLI: only "
                    "user and tool messages can"
                )
        return new, False


def _tool_specs(tools: list[tools_.Tool]) -> list[dict[str, Any]]:
    return [t.model_dump(mode="json") for t in tools]
