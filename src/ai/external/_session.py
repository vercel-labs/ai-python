"""`Session`: one session of an external harness's CLI.

The state is here: `id` and `messages`. So is the low level: `start`,
`send`, `events`, `interrupt` and `close`. The model protocol
(`ExternalHarnessProvider.stream`) is built on it.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import contextvars
import dataclasses
import functools
import json
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Self

import pydantic

from .. import experimental_telemetry as telemetry
from ..types import events as events_
from ..types import messages as messages_
from ..types import tools as tools_
from ..types import usage as usage_
from . import _bridge, _wire
from . import _errors as errors

if TYPE_CHECKING:
    from . import _harness

# Seconds to wait for more `ai` tool calls before the message ends: a CLI
# that calls tools in parallel sends the calls one by one.
BATCH_WINDOW = 0.05
# Seconds `interrupt()` waits for the CLI to end the turn before it closes it.
INTERRUPT_TIMEOUT = 30.0
# Key in `provider_metadata` for what the external harness records about a
# message.
METADATA_KEY = "external"

NEW_MESSAGE = "No result: a new user message came before the result."
INTERRUPTED = "No result: the turn was interrupted."
CLOSED = "No result: the external harness closed."
NOT_OPEN = (
    "the session is not open: use `async with harness.session()` "
    "or `await session.start()`"
)


# The sessions that are open in this context, the newest last. A model call of
# an external harness runs on the external harness's open session, if the
# context has one.
_OPEN: contextvars.ContextVar[tuple[Session, ...]] = contextvars.ContextVar(
    "ai_external_open_sessions", default=()
)


class MessageEvent(events_.BaseEvent):
    """A complete message went into `session.messages`."""

    message: messages_.Message
    kind: Literal["external_message"] = "external_message"


ExternalHarnessEvent = events_.Event | MessageEvent


class ExternalHarnessSessionSpanData(pydantic.BaseModel):
    """One external harness session, from `start()` to `close()`."""

    kind: Literal["external_session"] = "external_session"
    harness: str
    model: str | None = None
    cwd: str
    session_id: str | None = None


class ExternalHarnessMessageSpanData(pydantic.BaseModel):
    """One assistant message of an external harness."""

    kind: Literal["external_message"] = "external_message"
    harness: str
    model: str | None = None
    session_id: str | None = None
    message: messages_.Message | None = None
    usage: usage_.Usage | None = None
    finish_reason: str | None = None


class Session:
    """One session of an external harness's CLI: its state and direct control.

    `harness.session()` gives the external harness's session
    (`harness.session_id`), or a new one if the external harness has none.
    `harness.session(session_id)` continues another session that the CLI has,
    and `harness.session(messages=history)` makes a new one that starts with a
    past. The state is plain data:

    - `id`: the CLI's id for the session. None until a new session starts.
    - `messages`: the conversation, in `ai` messages. The session keeps it
      current.

    Open the session to keep the CLI alive between turns and to control it::

        async with harness.session() as session:
            await session.send(ai.user_message("Fix the tests"))
            async for event in session.events():
                ...

    `start()` and `close()` do the same without a block. While the session is
    open, the external harness's model calls (`ai.stream(harness, messages)`,
    `ai.Agent(...).run(harness, messages)`) run on it. This applies to the
    task that opened the session and to the tasks that it makes after that.

    The CLI starts with the first `send()` or the first model call, not when
    the block opens: by then the tools and the past are known. `start()`
    starts it at once.

    `provider_metadata["external"]` marks a message that the external harness
    wrote outside a complete model call: `{"stop_reason": "interrupted"}` on a
    reply that an interrupt or a close cut, `{"error": text}` on the reply of a
    failed turn, and `{"synthetic": True}` on a tool message for calls that got
    no result.
    """

    def __init__(
        self,
        harness: _harness.ExternalHarness,
        session_id: str | None = None,
        messages: list[messages_.Message] | None = None,
    ) -> None:
        self.harness = harness
        self.id = session_id if session_id is not None else harness.session_id
        self.messages = list(messages or [])
        # Open: the CLI stays alive between turns, and `send()` works.
        self._open = False
        self._run: _CliRun | None = None
        # What `events()` did not give yet.
        self._pending_events: collections.deque[Any] = collections.deque()
        self._pending_events_wake = asyncio.Event()
        # `events()` gave something since the last turn began.
        self._pending_events_read = False
        # The queues of the model calls that read. See the provider's
        # `stream()`.
        self._stream_queues: set[asyncio.Queue[Any]] = set()
        self._start_lock = asyncio.Lock()
        # Held by the provider's `stream()`: one model call at a time.
        self._stream_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        await self._mark_open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _mark_open(self) -> None:
        """Open the session: the external harness's model calls run on it."""
        if self._open:
            return
        opened = [s for s in _OPEN.get() if s._open]
        if self.id is not None and any(s.id == self.id for s in opened):
            raise errors.ExternalHarnessError(
                f"the session {self.id} is open already"
            )
        # The session that the external harness's model calls made without a
        # block.
        last = self.harness._session
        if last is not None and last.id == self.id:
            if last._stream_lock.locked():
                raise errors.ExternalHarnessError(
                    "a model call runs on this session: wait for its end"
                )
            self.harness._session = None
        self._open = True
        _OPEN.set((*opened, self))
        if last is not None and last.id == self.id:
            # Its CLI can wait in a turn that nobody continues.
            await last._stop_cli()

    async def start(self) -> None:
        """Open the session and start the CLI now. `close()` ends both.

        With `id`, continue that CLI session. Without it, start a new one,
        seeded with `messages` (lossy). Started already: nothing happens.
        After the CLI stopped, start it again.
        """
        await self._mark_open()
        await self._start()

    async def _start(self) -> None:
        async with self._start_lock:
            if self._run is not None and self._run.stopped is None:
                return
            if self._run is not None:
                run, self._run = self._run, None
                await run.close()
            run = _CliRun(self, self.harness.provider._make_wire(self.harness))
            await run.start()
            self._run = run

    async def send(self, message: messages_.Message) -> None:
        """Give a message to the CLI.

        A user message starts a turn, or steers the turn that runs. The CLI
        decides. A tool message gives results to `ai` tool calls that wait.
        `send()` returns when the CLI has the message; the message goes into
        `messages` when the CLI accepts it.
        """
        if message.role not in ("user", "tool"):
            raise ValueError(
                f"send() takes a user or tool message, not {message.role!r}"
            )
        if not self._open:
            raise errors.ExternalHarnessError(NOT_OPEN)
        await self._send(message)

    async def _send(self, message: messages_.Message) -> None:
        if self._run is None:
            await self._start()
        run = self._run
        if run is None:
            raise errors.ExternalHarnessError(
                "the session closed during send()"
            )
        if run.stopped is not None:
            raise errors.ExternalHarnessError(
                f"the external harness stopped: {run.stopped}"
            )
        if message.role == "tool":
            await run.run_in_pump(
                functools.partial(run.send_tool_results, message)
            )
            return
        await run.run_in_pump(functools.partial(run.fail_calls, NEW_MESSAGE))
        if run.idle.is_set():
            # A new turn. Nobody read `events()` during the last one: nobody
            # reads, so the feed must not grow.
            if not self._pending_events_read:
                self._pending_events.clear()
            self._pending_events_read = False
        run.idle.clear()
        run.used = True
        try:
            await run.wire.send(message)
        except BaseException:
            if run.hydrator is None and not run.waiting_calls:
                run.idle.set()
            raise

    async def events(self) -> AsyncGenerator[ExternalHarnessEvent]:
        """Read the session's events from where the last reader stopped.

        The session keeps the events that nobody read. Thus `send()` and then
        `events()` loses nothing, and a reader that stops needs no cleanup:
        the next `events()` continues. One reader: with two, each event goes
        to one of them. It ends when the session closes, and raises if the
        CLI fails.

        If nobody reads during a whole turn, the events before the latest
        turn are dropped. The messages stay in `messages`.
        """
        if not self._open and not self._pending_events:
            raise errors.ExternalHarnessError(NOT_OPEN)
        while True:
            while self._pending_events:
                item = self._pending_events.popleft()
                if isinstance(item, BaseException):
                    raise item
                self._pending_events_read = True
                yield item
            if not self._open:
                return
            self._pending_events_wake.clear()
            await self._pending_events_wake.wait()

    async def interrupt(self) -> None:
        """Stop the current turn. The CLI keeps the session.

        Return when the CLI ended the turn. If it does not end the turn in
        `INTERRUPT_TIMEOUT` seconds, stop the CLI; the next `send()` or model
        call starts it again. No turn: nothing happens.
        """
        run = self._run
        if run is None or run.stopped is not None:
            return
        await run.run_in_pump(functools.partial(run.fail_calls, INTERRUPTED))
        if run.idle.is_set():
            return
        run.interrupting = True
        try:
            async with asyncio.timeout(INTERRUPT_TIMEOUT):
                await run.wire.interrupt()
                await run.idle.wait()
        except TimeoutError:
            await self._stop_cli()

    async def close(self) -> None:
        """Stop the CLI and close the session. `id` and `messages` stay.

        A second close does nothing. `start()` opens the session again.
        """
        await self._stop_cli()
        self._open = False
        _OPEN.set(tuple(s for s in _OPEN.get() if s is not self))
        self._pending_events_wake.set()

    async def _stop_cli(self) -> None:
        """Stop the CLI. The session stays as it is, open or not."""
        async with self._start_lock:
            run, self._run = self._run, None
            if run is not None:
                await run.close()
            # A model call that got the end takes nothing after it.
            for queue in self._stream_queues:
                queue.put_nowait(_END)
            self._stream_queues.clear()

    def _publish_event(self, item: Any) -> None:
        """Give an event, or an error, to `events()` and to the model calls."""
        self._pending_events.append(item)
        self._pending_events_wake.set()
        for queue in self._stream_queues:
            queue.put_nowait(item)

    async def _start_for_stream(self, *, seeds: bool) -> _CliRun:
        """Start the CLI for a model call. For the provider's `stream()`.

        A CLI reads the tool list once, when it connects to the bridge, and
        takes a past only when its session starts. `seeds`: the session just
        took a past. If the tools or the past changed and no turn runs, start
        the CLI again.
        """
        harness = self.harness
        run = self._run
        if run is not None and run.stopped is None:
            restart = (seeds and run.fresh) or (
                _dump_tool_specs(harness.tools) != run.served_tool_specs
            )
            if restart and run.idle.is_set() and not run.waiting_calls:
                async with self._start_lock:
                    self._run = None
                    await run.close()
                if run.fresh and not run.used:
                    # The session is new and empty: start a new one, which also
                    # takes the past. The external harness forgets the empty
                    # one.
                    if harness.session_id == self.id:
                        harness.session_id = None
                    self.id = None
            elif restart:
                warnings.warn(
                    "the tools changed during a turn: the CLI keeps the "
                    "tools it saw at the start until the external harness "
                    "restarts",
                    stacklevel=2,
                )
        await self._start()
        run = self._run
        if run is None:
            raise errors.ExternalHarnessError(
                "the session closed during stream()"
            )
        run.bridge.set_tools(harness.tools)
        return run

    def _sync(
        self, messages: list[messages_.Message]
    ) -> list[messages_.Message]:
        """Match a whole history with the conversation; return the new part.

        First use: the history up to its last user messages becomes the
        conversation. After that, the history must continue the
        conversation.
        """
        if not self.messages:
            # Without an id the past seeds the new CLI session; with one, the
            # CLI has it already.
            start = len(messages)
            while start and messages[start - 1].role == "user":
                start -= 1
            if start == len(messages):
                raise errors.TranscriptMismatchError(
                    "the last message must be a user message: an external "
                    "harness cannot continue tool calls that it did not make"
                )
            self.messages = list(messages[:start])
            return list(messages[start:])
        # The external harness wrote these outside a complete model call: the
        # caller does not always have them.
        own = {
            m.id
            for m in self.messages
            if METADATA_KEY in (m.provider_metadata or {})
        }
        known = [m.id for m in self.messages if m.id not in own]
        given = [m for m in messages if m.id not in own]
        if [m.id for m in given[: len(known)]] != known:
            raise errors.TranscriptMismatchError(
                "the messages do not continue this session's conversation; "
                "use session.messages as the history, or a new session"
            )
        new = given[len(known) :]
        if not new:
            raise errors.TranscriptMismatchError(
                "the messages have nothing new for this session: "
                "add a user message"
            )
        for message in new:
            if message.role not in ("user", "tool"):
                raise errors.TranscriptMismatchError(
                    f"a new {message.role} message cannot go to the "
                    "CLI: only user and tool messages can"
                )
        return new


@dataclasses.dataclass
class _Call:
    """An `ai` tool call that the CLI made through the bridge. The CLI waits."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    answer: asyncio.Future[tuple[Any, bool]]


@dataclasses.dataclass
class _Command:
    """Work for the pump: only the pump task changes the conversation."""

    run: Callable[[], Awaitable[None]]
    done: asyncio.Future[None]


@dataclasses.dataclass(frozen=True)
class _Stopped:
    """The wire stopped: the CLI exited or failed."""

    error: BaseException


_END = object()  # put on a model call's queue when the CLI stops


class _CliRun:
    """One run of the CLI process: the wire, the bridge and the pump.

    The pump is the only task that changes the conversation state. Other
    tasks give it items through `queue`: wire items, bridge calls, commands.
    What it makes goes to the session's readers.
    """

    def __init__(self, session: Session, wire: _wire.Wire) -> None:
        self.session = session
        self.harness = harness = session.harness
        self.wire = wire
        self.bridge = _bridge.Bridge(self._handle_bridge_call)
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.tasks: list[asyncio.Task[None]] = []
        self.span: telemetry.Span[ExternalHarnessSessionSpanData] = (
            telemetry.create_span(
                ExternalHarnessSessionSpanData(
                    harness=harness.harness_class_id,
                    model=harness.model_name,
                    cwd=harness.cwd,
                    session_id=session.id,
                ),
                set_as_current=False,
            )
        )
        # Set when no turn runs. `interrupt()` waits on it.
        self.idle = asyncio.Event()
        self.idle.set()
        self.interrupting = False
        self.stopped: BaseException | None = None

        # For the provider's `stream()` only. The run records these; only
        # `stream()` reads them, to match the history it gets with this run.
        # A new CLI session, not one that continues `session_id`.
        self.fresh = session.id is None
        # The tools the CLI saw when it connected to the bridge.
        self.served_tool_specs: list[dict[str, Any]] = []
        # True after the first user message went to the CLI.
        self.used = False
        # Turn errors, by the id of the assistant message they ended.
        # `stream()` raises them; `events()` gives only the message.
        self.failures: dict[str, BaseException] = {}

        # The open assistant message.
        self.hydrator: events_.MessageHydrator | None = None
        self.message_id = ""
        self.message_span: (
            telemetry.Span[ExternalHarnessMessageSpanData] | None
        ) = None
        self.tool_spans: dict[
            str, telemetry.Span[telemetry.ToolExecutionSpanData]
        ] = {}
        # Text and reasoning blocks and CLI tool calls that did not end yet.
        # A steer waits for none to be open, so no block is cut in two.
        self.open: set[str] = set()
        # User messages that the CLI took while the message was busy.
        self.inputs: list[messages_.Message] = []
        # `ai` tool calls without a result. While there are some, other items
        # wait in `held_items`: they come after the results.
        self.waiting_calls: dict[str, _Call] = {}
        self.held_items: list[Any] = []
        # When the message with `ai` tool calls ends, if no call comes first.
        self.close_at: float | None = None

    # -- life -----------------------------------------------------------------

    async def start(self) -> None:
        harness = self.harness
        self.span.stamp_start()
        await self.span.push()
        await self.bridge.start()
        self.bridge.set_tools(harness.tools)
        self.served_tool_specs = _dump_tool_specs(harness.tools)
        session = self.session
        system = [m.text for m in session.messages if m.role == "system"]
        instructions = "\n\n".join(
            t for t in (harness.instructions, *system) if t
        )
        seed = (
            [m for m in session.messages if m.role != "system"]
            if session.id is None
            else []
        )
        try:
            session.id = await self.wire.start(
                session_id=session.id,
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
        if self.fresh and harness.session_id is None:
            # The external harness's first new session: the external harness
            # remembers it.
            harness.session_id = session.id
        self.span.data.session_id = session.id
        self.tasks = [
            asyncio.create_task(self._read_wire()),
            asyncio.create_task(self._run_pump()),
        ]

    async def close(self) -> None:
        if self.stopped is None:
            # End what is open, so the state stays valid.
            with contextlib.suppress(Exception):
                await self.run_in_pump(self._end_open_work)
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(BaseException):
                await task
        with contextlib.suppress(Exception):
            await self.wire.close()
        with contextlib.suppress(Exception):
            await self.bridge.close()
        self.stopped = self.stopped or errors.ExternalHarnessError(
            "the external harness is closed"
        )
        self.idle.set()
        self._fail_commands()
        await self.span.stamp_end().push()

    async def _end_open_work(self) -> None:
        await self.fail_calls(CLOSED)
        if self.hydrator is not None:
            await self._end_message("other", interrupted=True)

    async def run_in_pump(self, run: Callable[[], Awaitable[None]]) -> None:
        """Run `run` in the pump task and wait for it."""
        if self.stopped is not None:
            raise errors.ExternalHarnessError(
                f"the external harness stopped: {self.stopped}"
            )
        done = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(_Command(run, done))
        await done

    def _fail_commands(self) -> None:
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if isinstance(item, _Command) and not item.done.done():
                item.done.set_exception(
                    errors.ExternalHarnessError(
                        f"the external harness stopped: {self.stopped}"
                    )
                )

    # -- tasks ----------------------------------------------------------------

    async def _read_wire(self) -> None:
        error: BaseException
        try:
            async for item in self.wire.events():
                self.queue.put_nowait(item)
            error = errors.ExternalHarnessError(
                f"{self.harness.harness_class_id}: the CLI stopped"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        self.queue.put_nowait(_Stopped(error))

    async def _run_pump(self) -> None:
        while True:
            try:
                async with asyncio.timeout_at(self.close_at):
                    item = await self.queue.get()
            except TimeoutError:
                self.close_at = None
                if self.hydrator is not None and self.waiting_calls:
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
                await self._handle_item(item)
            except Exception as exc:
                await self._mark_stopped(exc)
                return

    async def _mark_stopped(self, error: BaseException) -> None:
        if not isinstance(error, errors.ExternalHarnessError):
            wrapped = errors.ExternalHarnessError(
                f"{self.harness.harness_class_id} failed: {error}"
            )
            wrapped.__cause__ = error
            error = wrapped
        for call in self.waiting_calls.values():
            if not call.answer.done():
                call.answer.set_result((str(error), True))
        self.waiting_calls.clear()
        if self.hydrator is not None:
            with contextlib.suppress(Exception):
                await self._end_message("error", error=error)
        self.stopped = error
        self.idle.set()
        self._fail_commands()
        self.session._publish_event(error)
        self.session._stream_queues.clear()

    async def _handle_bridge_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[Any, bool]:
        """Report a bridge call to an `ai` tool; wait for its result."""
        if self.stopped is not None:
            return f"the external harness stopped: {self.stopped}", True
        call = _Call(
            tool_call_id=messages_.generate_id("call"),
            name=name,
            arguments=arguments,
            answer=asyncio.get_running_loop().create_future(),
        )
        self.queue.put_nowait(call)
        return await call.answer

    # -- the conversation (pump task only) ------------------------------------

    async def _handle_item(self, item: Any) -> None:
        if self.waiting_calls and (
            self.hydrator is None or not isinstance(item, _Call)
        ):
            # Waiting for `ai` tool results: what comes now goes after them.
            if self.hydrator is not None:
                await self._end_message("tool_call")
            self.held_items.append(item)
            return
        match item:
            case _Call():
                await self._handle_tool_call(item)
            case _wire.Accepted(message=message):
                await self._handle_accepted(message)
            case _wire.TurnEnd():
                await self._handle_turn_end(item)
            case _:
                await self._handle_content(item)

    async def _handle_content(self, event: _wire.Content) -> None:
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

    async def _handle_accepted(self, message: messages_.Message) -> None:
        self.idle.clear()
        if self.hydrator is None:
            await self._append_message(message)
            return
        # A steer: the assistant message ends, the user message follows,
        # and a new assistant message starts.
        self.inputs.append(message)
        if not self.open:
            await self._end_message("other")

    async def _handle_tool_call(self, call: _Call) -> None:
        if self.hydrator is None:
            await self._begin_message()
        self.waiting_calls[call.tool_call_id] = call
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

    async def _handle_turn_end(self, end: _wire.TurnEnd) -> None:
        interrupted = end.error is None and (
            end.reason == "interrupted" or self.interrupting
        )
        reason: str = "other" if interrupted else end.reason
        # Every turn ends with one StreamEnd, also a turn that said nothing.
        await self._end_message(
            reason,
            usage=end.usage,
            interrupted=interrupted,
            error=end.error,
            metadata=end.metadata,
        )
        self.interrupting = False
        self.idle.set()

    async def send_tool_results(self, message: messages_.Message) -> None:
        """Give `ai` tool results to the CLI calls that wait for them."""
        results = message.tool_results
        if not results:
            raise ValueError("a tool message needs at least one tool result")
        unknown = [
            p.tool_call_id
            for p in results
            if p.tool_call_id not in self.waiting_calls
        ]
        if unknown:
            raise ValueError(f"no tool call waits for these results: {unknown}")
        if self.hydrator is not None:
            await self._end_message("tool_call")
        for part in results:
            call = self.waiting_calls.pop(part.tool_call_id)
            value = part.model_input
            if value is messages_.MODEL_INPUT_UNSET:
                value = part.result
            if not call.answer.done():
                call.answer.set_result((value, part.result_kind == "error"))
        await self._append_message(message)
        if not self.waiting_calls:
            await self._handle_held_items()

    async def fail_calls(self, reason: str) -> None:
        """Answer the waiting `ai` tool calls with an error."""
        if not self.waiting_calls:
            return
        if self.hydrator is not None:
            await self._end_message("tool_call")
        parts: list[messages_.Part] = []
        for call in self.waiting_calls.values():
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
        self.waiting_calls.clear()
        # A tool message must follow the calls, or no model takes the history.
        await self._append_message(
            messages_.Message(
                role="tool",
                parts=parts,
                provider_metadata={METADATA_KEY: {"synthetic": True}},
            )
        )
        await self._handle_held_items()

    async def _handle_held_items(self) -> None:
        held, self.held_items = self.held_items, []
        for item in held:
            await self._handle_item(item)

    async def _begin_message(self) -> None:
        harness = self.harness
        self.message_id = messages_.generate_id("msg")
        self.hydrator = events_.MessageHydrator()
        self.open.clear()
        self.message_span = telemetry.create_span(
            ExternalHarnessMessageSpanData(
                harness=harness.harness_class_id,
                model=harness.model_name,
                session_id=self.session.id,
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.hydrator is None:
            await self._begin_message()
        assert self.hydrator is not None
        self.close_at = None
        metadata = dict(metadata or {})
        if error is not None:
            metadata[METADATA_KEY] = {"error": str(error)}
            self.failures[self.message_id] = error
        elif interrupted:
            # Not in the finish-reason vocabulary: `other`, and the raw value.
            metadata[METADATA_KEY] = {"stop_reason": "interrupted"}
        self._emit(
            events_.StreamEnd(
                finish_reason=reason,
                usage=usage,
                provider_metadata=metadata or None,
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
        await self._append_message(message)
        inputs, self.inputs = self.inputs, []
        if reason == "tool_call":
            # The tool results must follow the calls: steers wait for them.
            self.held_items[:0] = [_wire.Accepted(m) for m in inputs]
        else:
            for m in inputs:
                await self._append_message(m)

    def _emit(self, event: events_.Event) -> None:
        assert self.hydrator is not None
        stamp = messages_.Message(
            id=self.message_id, role="assistant", parts=[]
        )
        self.session._publish_event(
            self.hydrator.feed(event.model_copy(update={"message": stamp}))
        )

    async def _append_message(self, message: messages_.Message) -> None:
        # A new list, not append: a caller can hold the old one.
        self.session.messages = [*self.session.messages, message]
        self.session._publish_event(MessageEvent(message=message))


def _dump_tool_specs(tools: list[tools_.Tool]) -> list[dict[str, Any]]:
    return [t.model_dump(mode="json") for t in tools]
