"""A conversation, and the verbs that control it while it runs.

Control lives HERE, not on the turn: there is only ever one active turn per
session, so `steer`/`stop` need no handle to target. That is what keeps the
Turn a read-only stream and removes the entire class of bugs that came from
an object which was both awaitable and iterable.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from ...types import events
from ...types import messages as messages_
from ...types import usage as usage_
from . import _handle, _input, _result, errors
from . import _structured as _structured_

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...workspaces.experimental import _pty
    from . import _harness

T = TypeVar("T")


class SessionInfo(BaseModel):
    """One conversation in the harness's own store.

    `kind` names the harness that owns it. A session id alone does not:
    codex ids happen to be UUIDv7 and claude's UUIDv4, which is an accident
    of two implementations, not something to build on. Gather sessions
    from several harnesses into one list and each still says whose it is —
    the same field `Handle.kind` carries, which is how `resume()` picks the
    adapter.
    """

    kind: str
    session_id: str
    title: str | None = None
    cwd: str | None = None
    updated_at: int | None = None
    created_at: int | None = None
    running: bool = False
    """Whether a process has this conversation open. A client of this SDK or
    an `agent.tui()` holds it, and `resume` refuses with SessionBusyError;
    the CLI started in a terminal is seen too (any `claude`, a `codex` only
    as `codex resume <id>`) but holds nothing `resume` checks. `fork` it
    instead."""
    pid: int | None = None
    """The process that has it open, in the workspace, when known."""


STOP_TIMEOUT = 30.0
"""Seconds `stop()` waits for an interrupted turn to settle before it returns
anyway. A turn whose CLI ignores the interrupt is the CLI's problem; the
caller of `stop()` must not hang on it."""


class Turn:
    """A settling turn: iterate it for events, then read `result`."""

    def __init__(
        self, session: Session, prompt: list[_input.Part], sent: str
    ) -> None:
        self._session = session
        self._prompt = prompt
        self._sent = sent
        self._result: _result.Result | None = None
        self._consumed = False
        self._inner: Any = None
        # Set once `_drive` has settled the turn, whichever task settled it.
        self._done = asyncio.Event()
        # The one close of `_inner`, shared by everyone who asks for it:
        # closing an async generator while another close of it is still
        # running raises, so a second caller waits on the first.
        self._closer: asyncio.Future[None] | None = None

    @property
    def result(self) -> _result.Result:
        if self._result is None:
            raise RuntimeError(
                "turn is not settled yet — iterate it to completion first"
            )
        return self._result

    @property
    def settled(self) -> bool:
        return self._result is not None

    async def __aiter__(self) -> AsyncIterator[events.AgentEvent]:
        if self._consumed:
            # A consumed stream is exhausted, never silently replayed.
            return
        self._consumed = True
        self._inner = self._session._drive(self)
        try:
            async for event in self._inner:
                yield event
        finally:
            # Reached on `break` too: Python closes this generator once the
            # loop that dropped it is gone, and that close is what ends a
            # turn nobody is reading, rather than leaving the agent working
            # until some later turn notices.
            await self._finish(wait=False)

    async def aclose(self) -> None:
        """Close the stream and settle now.

        A turn waiting at an event is closed on the spot. A turn another
        task is reading right now cannot be closed from here, so this waits
        for it to settle instead.
        """
        await self._finish(wait=True)

    async def _finish(
        self, *, wait: bool, timeout: float | None = None
    ) -> None:
        """Settle the turn: close it if it is paused, else maybe wait.

        An async generator can only be closed while it is suspended at a
        `yield`. While a consumer is awaiting its next event it is RUNNING,
        and closing it raises — so then the adapter must end the turn (after
        `adapter.stop()`, it does) and this waits for that, up to `timeout`.
        """
        inner = self._inner
        if inner is None or self._done.is_set():
            return
        if self._closer is None and not inner.ag_running:
            self._closer = asyncio.ensure_future(inner.aclose())
        if self._closer is not None:
            await asyncio.shield(self._closer)
            return
        if not wait:
            return
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(timeout):
                await self._done.wait()


class Tui:
    """A conversation a human is driving, in the harness's own TUI.

    The sibling of `Session`: two ways a conversation runs, two types with
    the same identity and the same two lifecycle verbs, and different
    capabilities. A `Session` is driven by this API (`run`, `steer`,
    `stop`); a `Tui` is driven by whoever `tty.bridge` connects, so it
    carries no such verbs — calling `run()` on a TUI conversation would
    open a second writer on it, and the lock would refuse.

    `session_id` is known at launch when the CLI let us choose it (claude)
    or told us (codex, for a conversation with a past). A bare `tui()` on
    codex never learns it: codex picks its own id, so it stays None; find
    the conversation with `sessions()`.
    `handle` is what an app saves to come back later, like `session.handle`.
    """

    def __init__(
        self, harness: _harness.Harness, pty: _pty.Pty, session_id: str | None
    ) -> None:
        self._harness = harness
        self.pty = pty
        self.session_id = session_id

    @property
    def handle(self) -> _handle.Handle | None:
        if self.session_id is None:
            return None
        return _handle.Handle(
            kind=self._harness.kind,
            session_id=self.session_id,
            workspace=self._harness.workspace.coords,
            options=self._harness.options,
            harness_version=self._harness.version,
        )

    async def detach(self) -> None:
        """Drop this connection.

        A named TUI keeps running — and keeps its claim on the conversation;
        attach again with `workspace.attach`.
        """
        await self.pty.detach()

    async def close(self) -> None:
        """End the TUI, then release the conversation's claim."""
        await self.pty.close()
        if self.session_id is not None:
            await self._harness.adapter.release_tui_lock(self.session_id)

    def __repr__(self) -> str:
        return f"<Tui {self.session_id or '?'} on {self._harness.name}>"


class Session:
    def __init__(
        self,
        harness: _harness.Harness,
        session_id: str,
        history: list[messages_.Message] | None = None,
    ) -> None:
        self._harness = harness
        self._session_id = session_id
        # A history to begin FROM: stored natively by the harness when the
        # session is created, and this session's own past from message one.
        self._history: list[messages_.Message] = list(history or [])
        self._messages: list[messages_.Message] = list(self._history)
        self._lock = asyncio.Lock()
        self._active: Turn | None = None
        self._stopped = False
        # The agent has ended the running turn (its StreamEnd arrived), even
        # if the consumer has not read to the end yet.
        self._ended = False
        # The assistant message the running turn is writing into, and the
        # hydrator that builds it; see `_drive`.
        self._segment = messages_.Message(role="assistant", parts=[])
        self._hydrator = events.MessageHydrator()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> list[messages_.Message]:
        """The conversation so far, in AI SDK shape."""
        return list(self._messages)

    @property
    def handle(self) -> _handle.Handle:
        return _handle.Handle(
            kind=self._harness.kind,
            session_id=self._session_id,
            workspace=self._harness.workspace.coords,
            options=self._harness.options,
            harness_version=self._harness.version,
        )

    # -- turns ----------------------------------------------------------------

    async def run(
        self,
        prompt: _input.Prompt,
        *,
        output_type: type[T] | None = None,
        retries: int = 1,
        timeout: float | None = None,
    ) -> _result.Result:
        """Send a prompt and wait for the turn to settle.

        `prompt` is a string, or a list of strings, `TextPart`s and
        `FilePart`s (images; PDFs where `capabilities.documents`; text files).
        """
        if output_type is None:
            return await self._once(prompt, timeout=timeout)
        return await self._structured(prompt, output_type, retries, timeout)

    def stream(self, prompt: _input.Prompt) -> Turn:
        """Send a prompt and iterate its events. Read `.result` after.

        A file the harness cannot take raises `UnsupportedError` here,
        before anything runs.
        """
        parts = _input.parts(prompt)
        _input.check(parts, self._harness.capabilities, self._harness.name)
        return Turn(self, parts, _input.text(parts))

    async def reclaim(self) -> None:
        """Settle a turn whose consumer walked away. Rarely needed by hand.

        `run()` and `stream()` already do this before starting a turn.
        """
        await self._reclaim()

    async def _reclaim(self) -> None:
        """Settle a turn whose consumer walked away.

        `async for event in turn: break` is ordinary Python, and it must not
        strand the session or leave the agent working. Closing the abandoned
        stream runs `_drive`'s cleanup: it stops the agent, settles the
        turn, and frees the lock for the next one. A turn another task is
        reading right now is left alone; the lock makes the next turn wait.
        """
        turn = self._active
        if turn is not None and not turn.settled:
            await turn._finish(wait=False)

    async def _once(
        self, prompt: _input.Prompt, *, timeout: float | None = None
    ) -> _result.Result:
        turn = self.stream(prompt)
        if timeout is None:
            async for _ in turn:
                pass
            return turn.result
        # A deadline is a settlement, not an exception: the turn is
        # cancelled where it stands, `_drive` stops the agent, and whatever
        # it produced is kept.
        try:
            async with asyncio.timeout(timeout):
                async for _ in turn:
                    pass
        except TimeoutError:
            if not turn.settled:
                return _result.Result(
                    finish_reason="cancelled",
                    messages=self.messages,
                    prompt_sent=turn._sent,
                )
            return turn.result.model_copy(update={"finish_reason": "cancelled"})
        return turn.result

    async def _structured(
        self,
        prompt: _input.Prompt,
        output_type: type[T],
        retries: int,
        timeout: float | None,
    ) -> _result.Result:
        """Run a turn and parse its answer as `output_type`.

        Structured output is a LAYER over run(), never inside the turn driver:
        it only ever calls `_once`, so the settling path knows nothing about
        schemas.
        """
        # The schema goes into the prompt's text: onto its last text part,
        # so a plain string prompt reads exactly as before.
        parts = _input.parts(prompt)
        last = parts[-1]
        if isinstance(last, messages_.TextPart):
            parts[-1] = messages_.TextPart(
                text=_structured_.annotate(last.text, output_type)
            )
        else:
            parts.append(
                messages_.TextPart(
                    text=_structured_.annotate("", output_type).lstrip()
                )
            )
        sent = _input.text(parts)
        result = await self._once(parts, timeout=timeout)
        result.prompt_sent = sent
        for attempt in range(retries + 1):
            parsed, error = _structured_.parse(result.text, output_type)
            if error is None:
                result.output = parsed
                return result
            if attempt == retries or result.cancelled:
                raise error
            result = await self._once(
                f"Your previous reply failed validation: {error}\n"
                + "Reply again with ONLY the corrected JSON.",
                timeout=timeout,
            )
            result.prompt_sent = sent
        return result

    async def _ensure(self) -> None:
        """Create the harness-side conversation on first use."""
        if not self._session_id:
            self._session_id = await self._harness.adapter.new_session(
                history=self._history or None
            )
            self._harness._register(self)

    async def _drive(self, turn: Turn) -> AsyncIterator[events.AgentEvent]:
        """One turn at a time; the turn stays open for writing while it runs.

        The lock is acquired manually rather than with `async with`, because
        an abandoned async generator only runs its cleanup when it is
        CLOSED. Holding the lock across yields inside a context manager
        meant that `async for event in turn: break` — ordinary Python —
        wedged the session forever. `_reclaim()` closes such a turn.
        """
        await self._ensure()
        # A turn nobody is reading any more is ended before this one starts.
        await self._reclaim()
        await self._lock.acquire()
        self._active = turn
        self._stopped = False
        self._ended = False
        settled: events.StreamEnd | None = None
        abandoned = False
        # The AI SDK reconstructs messages from the event stream, so a turn
        # abandoned mid-flight still carries whatever arrived — that is what
        # makes an interruption a settlement rather than a loss.
        hydrator = events.MessageHydrator()
        # History is the SESSION's, single-sourced: the prompt is a real
        # message in the transcript before the agent ever sees it.
        self._messages.append(
            messages_.Message(role="user", parts=list(turn._prompt))
        )
        start = len(self._messages)
        # Message identity is decided HERE, not by adapters. Every model
        # event is stamped with the current assistant SEGMENT, and a tool
        # result (or a steer) starts a new one, exactly as the AI SDK's own
        # agent loop does. A segment joins the history the moment it has
        # content, so history always reads in the order things happened —
        # call, result, answer — mid-turn included, which is what an
        # approval hook sees. The adapter's own StreamEnd is kept as `raw`
        # for settlement; only the streamed copy is re-identified.
        self._segment = messages_.Message(role="assistant", parts=[])
        self._hydrator = hydrator
        appended: set[str] = set()
        inner = self._harness.adapter.turn(self._session_id, turn._prompt)
        try:
            async for raw in inner:
                fed = raw
                if isinstance(raw, events.ModelEvent):
                    fed = raw.model_copy(
                        update={
                            "message": messages_.Message(
                                id=self._segment.id, role="assistant", parts=[]
                            )
                        }
                    )
                # Events are frozen, so the hydrator cannot annotate them in
                # place: it returns a COPY whose `.message` is the running
                # assistant message, and whose `ToolEnd.tool_call` is the
                # complete call. That copy is what the AI SDK's contract
                # says a consumer receives — every event carries the
                # message so far.
                event = hydrator.feed(fed)
                if isinstance(raw, events.StreamEnd):
                    # Settle on the adapter's own final message: it decides
                    # what the answer is (codex leaves its preamble out).
                    settled = raw
                    self._ended = True
                elif isinstance(raw, events.ToolCallResult):
                    # Tool results are their own messages, after the call.
                    self._messages.append(raw.message)
                    self._segment = messages_.Message(
                        role="assistant", parts=[]
                    )
                else:
                    self._join_segment(appended)
                yield event
        except (GeneratorExit, asyncio.CancelledError):
            abandoned = True
            raise
        except errors.HarnessError as exc:
            # Which CLI failed matters most when it is a newer one than the
            # SDK was verified with: say so on every failure.
            exc.add_note(
                f"{self._harness.name} version: "
                + f"{self._harness.version or 'unknown'}"
            )
            raise
        finally:
            try:
                if abandoned and settled is None and not self._stopped:
                    # Nobody reads this turn any more — a `break`, a
                    # deadline, a cancelled task — but the agent is still
                    # working on it. End it there too, or its leftovers
                    # would become the next turn's answer.
                    self._stopped = True
                    with contextlib.suppress(Exception):
                        await self._harness.adapter.stop(self._session_id)
                # Close the adapter's turn too: walking away from it is not
                # enough to run its cleanup.
                aclose = getattr(inner, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()
            finally:
                self._active = None
                approval_errors = self._harness._take_approval_errors(
                    self._session_id
                )
                turn._result = self._settle(
                    settled, start, turn, approval_errors
                )
                turn._done.set()
                self._lock.release()

    def _join_segment(self, appended: set[str]) -> None:
        """Put the current assistant segment in history once it has content."""
        segment = self._hydrator.messages_by_id.get(self._segment.id)
        if segment is not None and segment.parts and segment.id not in appended:
            appended.add(segment.id)
            self._messages.append(segment)

    def _settle(
        self,
        end: events.StreamEnd | None,
        start: int,
        turn: Turn,
        approval_errors: list[str],
    ) -> _result.Result:
        # History already holds every segment that streamed, in order. The
        # answer is the adapter's own final message when it sent one; else
        # whatever the stream got to, across all of this turn's segments.
        mine = [m for m in self._messages[start:] if m.role == "assistant"]
        if end is not None and end.message.parts and not mine:
            # A turn whose content never streamed: the final message is all
            # there is.
            self._messages.append(end.message)
            mine = [end.message]
        elif end is not None and mine and end.message.provider_metadata:
            mine[-1].provider_metadata = end.message.provider_metadata
        if end is not None:
            text = _text_of(end.message)
        else:
            text = _text_of(
                messages_.Message(
                    role="assistant",
                    parts=[part for m in mine for part in m.parts],
                )
            )
        # A stopped turn is cancelled regardless of what the harness called
        # it: we are the ones who interrupted it.
        finish = (
            "cancelled"
            if self._stopped
            else (end.finish_reason if end else "cancelled")
        )
        return _result.Result(
            finish_reason=finish or "stop",
            text=text,
            messages=self.messages,
            usage=(
                end.usage if end is not None and end.usage else usage_.Usage()
            ),
            prompt_sent=turn._sent,
            approval_errors=approval_errors,
        )

    # -- control --------------------------------------------------------------

    async def steer(self, text: str) -> None:
        """Inject guidance into the turn that is running right now."""
        if not self._harness.capabilities.steer:
            raise errors.UnsupportedError(self._harness.name, "steer")
        if self._active is None or self._ended:
            # Ended means ended: a message sent now is not guidance for this
            # turn but the start of another one nobody is reading.
            raise RuntimeError("no active turn to steer")
        await self._harness.adapter.steer(self._session_id, text)
        # A steering message is a real user message. If it only existed on
        # the wire, the transcript would not explain why the agent changed
        # course. What the agent says after it is a new assistant message.
        self._messages.append(
            messages_.Message(
                role="user", parts=[messages_.TextPart(text=text)]
            )
        )
        self._segment = messages_.Message(role="assistant", parts=[])

    async def stop(self) -> None:
        """Interrupt gracefully, from anywhere: the loop body, another task.

        The turn settles as cancelled; the harness keeps its transcript and
        the conversation stays resumable. Returns once the turn has settled,
        so `turn.result` is readable right after — or after `STOP_TIMEOUT`
        seconds if the agent ignores the interrupt. No turn running: nothing
        to do.
        """
        if not self._harness.capabilities.stop:
            raise errors.UnsupportedError(self._harness.name, "stop")
        turn = self._active
        if turn is None or turn.settled:
            return
        self._stopped = True
        await self._harness.adapter.stop(self._session_id)
        await turn._finish(wait=True, timeout=STOP_TIMEOUT)

    async def close(self) -> None:
        await self._harness.adapter.close_session(self._session_id)
        self._harness._forget(self._session_id)

    def _seed(self, messages: list[messages_.Message]) -> None:
        """Install a replayed transcript as this session's past."""
        self._messages = list(messages)

    def __repr__(self) -> str:
        return f"<Session {self._session_id} on {self._harness.name}>"


def _text_of(message: messages_.Message | None) -> str:
    """Return the agent's FINAL message: what it said after its last action.

    A turn's assistant message holds every part in order — "I'll write the
    file", the tool call, "Done". The text a caller wants is the last of
    those, not the narration glued to it. Codex labels its preamble
    `commentary` and the adapter already drops it; claude does not label
    anything, so the boundary has to be found here, harness-agnostically:
    the last tool call splits the turn, and only what follows it is the
    answer. A turn with no tool calls is one segment, all of it answer.
    """
    if message is None:
        return ""
    parts = message.parts
    last_call = max(
        (
            i
            for i, part in enumerate(parts)
            if isinstance(part, messages_.ToolCallPart)
        ),
        default=-1,
    )
    return "".join(
        part.text
        for part in parts[last_call + 1 :]
        if isinstance(part, messages_.TextPart)
    )
