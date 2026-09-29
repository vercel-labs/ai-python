"""A conversation, and the verbs that control it while it runs.

Control lives HERE, not on the turn: there is only ever one active turn per
session, so `steer`/`stop` need no handle to target. That is what keeps the
Turn a read-only stream and removes the entire class of bugs that came from
an object which was both awaitable and iterable.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from ...types import events
from ...types import messages as messages_
from ...types import usage as usage_
from . import _handle, _result, errors
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
    """Whether a client currently holds this conversation open — it would
    refuse `resume` with SessionBusyError; `fork` it instead."""


class Turn:
    """A settling turn: iterate it for events, then read `result`."""

    def __init__(self, session: Session, prompt: str, sent: str) -> None:
        self._session = session
        self._prompt = prompt
        self._sent = sent
        self._result: _result.Result | None = None
        self._consumed = False
        self._inner: Any = None

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
        async for event in self._inner:
            yield event

    async def aclose(self) -> None:
        """Close the stream and settle now.

        Cancelling a consumer does NOT run an async generator's `finally`
        promptly — cleanup waits for close or garbage collection — so a
        deadline must close the turn explicitly or it would settle with
        nothing and lose everything the harness already streamed.
        """
        if self._inner is not None:
            await self._inner.aclose()
            self._inner = None


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
        prompt: str,
        *,
        output_type: type[T] | None = None,
        retries: int = 1,
        timeout: float | None = None,
    ) -> _result.Result:
        """Send a prompt and wait for the turn to settle."""
        if output_type is None:
            return await self._once(prompt, timeout=timeout)
        return await self._structured(prompt, output_type, retries, timeout)

    def stream(self, prompt: str) -> Turn:
        """Send a prompt and iterate its events. Read `.result` after."""
        return Turn(self, prompt, prompt)

    async def reclaim(self) -> None:
        """Public form of `_reclaim`, for callers driving turns by hand."""
        await self._reclaim()

    async def _reclaim(self) -> None:
        """Settle a turn whose consumer walked away.

        `async for event in turn: break` is ordinary Python, and it must not
        strand the session. Closing the abandoned stream runs its cleanup,
        settles it, and frees the lock for the next turn.
        """
        turn = self._active
        if turn is not None and not turn.settled:
            await turn.aclose()

    async def _once(
        self, prompt: str, *, timeout: float | None = None
    ) -> _result.Result:
        await self._ensure()
        await self._reclaim()
        turn = Turn(self, prompt, prompt)
        if timeout is None:
            async for _ in turn:
                pass
            return turn.result
        # A deadline is a settlement, not an exception: stop the agent
        # gracefully and keep whatever it produced.
        try:
            async with asyncio.timeout(timeout):
                async for _ in turn:
                    pass
        except TimeoutError:
            await self._harness.adapter.stop(self._session_id)
            self._stopped = True
            await turn.aclose()
            if not turn.settled:
                return _result.Result(
                    finish_reason="cancelled",
                    messages=self.messages,
                    prompt_sent=prompt,
                )
            return turn.result.model_copy(update={"finish_reason": "cancelled"})
        return turn.result

    async def _structured(
        self,
        prompt: str,
        output_type: type[T],
        retries: int,
        timeout: float | None,
    ) -> _result.Result:
        """Run a turn and parse its answer as `output_type`.

        Structured output is a LAYER over run(), never inside the turn driver:
        it only ever calls `_once`, so the settling path knows nothing about
        schemas.
        """
        sent = _structured_.annotate(prompt, output_type)
        result = await self._once(sent, timeout=timeout)
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
        await self._lock.acquire()
        self._active = turn
        self._stopped = False
        settled: events.StreamEnd | None = None
        # The AI SDK reconstructs messages from the event stream, so a turn
        # abandoned mid-flight still carries whatever arrived — that is what
        # makes an interruption a settlement rather than a loss.
        hydrator = events.MessageHydrator()
        # History is the SESSION's, single-sourced: the prompt is a real
        # message in the transcript before the agent ever sees it.
        self._messages.append(
            messages_.Message(
                role="user", parts=[messages_.TextPart(text=turn._prompt)]
            )
        )
        # Message identity is decided HERE, not by adapters. The hydrator
        # needs an id to know where a message ends: after a tool result it
        # switches to that role="tool" message, and an identity-less text
        # event that follows is glued onto it — then a StreamEnd carrying
        # some other id gets written onto the same object, which is already
        # in history. So every model event is stamped with the current
        # assistant segment, and a tool result starts a new segment, exactly
        # as the AI SDK's own agent loop does. The adapter's own StreamEnd
        # is kept as `raw` for settlement; only the streamed copy is
        # re-identified.
        segment = messages_.Message(role="assistant", parts=[])
        try:
            async for raw in self._harness.adapter.turn(
                self._session_id, turn._prompt
            ):
                fed = raw
                if isinstance(raw, events.ModelEvent):
                    fed = raw.model_copy(
                        update={
                            "message": messages_.Message(
                                id=segment.id, role="assistant", parts=[]
                            )
                        }
                    )
                # Events are frozen, so the hydrator cannot annotate them in
                # place: it returns a COPY whose `.message` is the running
                # assistant message, and whose `ToolEnd.tool_call` is the
                # complete call. That copy is what the AI SDK's contract
                # says a consumer receives — every event carries the
                # message so far. Discarding it, as this loop once did,
                # handed out events whose `.message` was an empty
                # placeholder until StreamEnd.
                event = hydrator.feed(fed)
                if isinstance(raw, events.StreamEnd):
                    # Settle on the adapter's own final message: it carries
                    # provider metadata the reconstruction does not.
                    settled = raw
                elif isinstance(raw, events.ToolCallResult):
                    # Tool results are their own messages; they land in
                    # history as they arrive, not at settle.
                    self._messages.append(raw.message)
                    segment = messages_.Message(role="assistant", parts=[])
                yield event
        finally:
            self._active = None
            approval_errors = self._harness._take_approval_errors()
            turn._result = self._settle(
                settled, hydrator, turn, approval_errors
            )
            self._lock.release()

    def _settle(
        self,
        end: events.StreamEnd | None,
        hydrator: events.MessageHydrator,
        turn: Turn,
        approval_errors: list[str],
    ) -> _result.Result:
        # Whatever the harness finished with, or whatever the stream got to
        # before it was interrupted. An interrupted turn may span several
        # assistant segments (one per tool round); keep all of them.
        if end is not None:
            message: messages_.Message | None = end.message
        else:
            parts = [
                part
                for m in hydrator.messages
                if m.role == "assistant"
                for part in m.parts
            ]
            message = messages_.Message(role="assistant", parts=parts)
        if message is not None and not message.parts:
            message = None
        if message is not None:
            self._messages.append(message)
        # A stopped turn is cancelled regardless of what the harness called
        # it: we are the ones who interrupted it.
        finish = (
            "cancelled"
            if self._stopped
            else (end.finish_reason if end else "cancelled")
        )
        return _result.Result(
            finish_reason=finish or "stop",
            text=_text_of(message),
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
        if self._active is None:
            raise RuntimeError("no active turn to steer")
        await self._harness.adapter.steer(self._session_id, text)
        # A steering message is a real user message. If it only existed on
        # the wire, the transcript would not explain why the agent changed
        # course.
        self._messages.append(
            messages_.Message(
                role="user", parts=[messages_.TextPart(text=text)]
            )
        )

    async def stop(self) -> None:
        """Interrupt gracefully.

        The turn settles as cancelled; the harness keeps its transcript and the
        conversation stays resumable.
        """
        if not self._harness.capabilities.stop:
            raise errors.UnsupportedError(self._harness.name, "stop")
        self._stopped = True
        await self._harness.adapter.stop(self._session_id)
        # Stopping means the turn is over: settle it now so `turn.result` is
        # readable and the session is immediately usable again.
        await self._reclaim()

    async def close(self) -> None:
        await self._harness.adapter.close_session(self._session_id)

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
