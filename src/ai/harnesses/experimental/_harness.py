"""The harness: one configured agent CLI, open on one workspace."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Self, TypeVar

from . import _capabilities, _result, _session
from . import errors as errors_

if TYPE_CHECKING:
    from ...types import messages
    from ...workspaces.experimental import _base
    from ._adapters import base

T = TypeVar("T")

DEFAULT_APPROVAL_TIMEOUT = 300.0


class Harness:
    def __init__(
        self,
        adapter: base.Adapter,
        *,
        workspace: _base.Workspace,
        approve: base.ApprovalHook | None = None,
        approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.workspace = workspace
        # No hook means every tool call is approved. The harness's own
        # default is to ask the person at the terminal, and in a program
        # that person does not exist — the agent would wait on a question
        # nobody answers. A hook is how you RESTRICT it.
        self.approve = approve
        self.approval_timeout = approval_timeout
        self.options = dict(options or {})
        self._open = False
        self._approval_errors: list[str] = []
        # Adapters do not keep their own transcript: they borrow the
        # session's, so the approval hook and the caller see one history.
        self._sessions: dict[str, Any] = {}

    # -- identity -------------------------------------------------------------

    @property
    def kind(self) -> str:
        return self.adapter.kind

    @property
    def name(self) -> str:
        return self.adapter.kind

    @property
    def version(self) -> str | None:
        return self.adapter.version

    @property
    def capabilities(self) -> _capabilities.Capabilities:
        return self.adapter.capabilities

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def process_ids(self) -> list[int]:
        return self.adapter.process_ids if self._open else []

    # -- lifecycle ------------------------------------------------------------

    async def open(self) -> None:
        """Probe the executable and prepare the workspace. Idempotent.

        Any failure past this point must unwind everything acquired: the
        caller's `async with` never ran, so nothing else will clean up.
        """
        if self._open:
            return
        await self.workspace.open()
        # The hook and its failure sink belong to the harness; the adapter
        # only needs to call them.
        self.adapter.approve = self.approve
        self.adapter.approval_timeout = self.approval_timeout
        self.adapter.on_approval_error = self._record_approval_error
        # NB: named `live_history`, not `history` — the adapter's own
        # history() reads STORED transcripts. Two different questions.
        self.adapter.live_history = self._history_of
        try:
            await self.adapter.start(self.workspace, self.options)
        except BaseException:
            await self._unwind()
            raise
        self._open = True

    async def _unwind(self) -> None:
        with contextlib.suppress(Exception):
            await self.adapter.close()

    async def close(self) -> None:
        if not self._open:
            return
        self._open = False
        await self.adapter.close()

    async def detach(self) -> None:
        """Release this client's hold, leaving the harness RUNNING.

        Release without tearing down: drop this client's connection and
        leave the harness running (in a sandbox opened with keep=True it
        persists with the VM). With no hook it runs on unattended; with one
        it blocks on the next tool call until a client reconnects. On a
        workspace whose processes die with the client, this is close().
        """
        if not self._open:
            return
        self._open = False
        await self.adapter.detach()

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- work -----------------------------------------------------------------

    def session(
        self, *, history: list[messages.Message] | None = None
    ) -> _session.Session:
        """Start a new conversation.

        The harness-side session is created on first use, so this stays
        synchronous and cheap.

        Deliberately NOT a Session subclass that wraps `_drive`: an extra
        generator in the chain means closing the outer one does not
        synchronously close the inner one, and the turn's cleanup — the code
        that settles it and frees the session — gets deferred to garbage
        collection.
        """
        if not self._open:
            raise errors_.HarnessClosedError(
                "harness is not open; call open() or use `async with`"
            )
        return _session.Session(self, "", history=history)

    async def run(
        self,
        prompt: str,
        *,
        history: list[messages.Message] | None = None,
        output_type: type[T] | None = None,
        retries: int = 1,
        timeout: float | None = None,
    ) -> _result.Result:
        """One prompt in a throwaway conversation.

        `history` starts that conversation from an earlier one — any
        `history()` or `Result.messages`, from either harness, from any
        machine. Sugar over `session(history=...).run(...)`.
        """
        if not self._open:
            raise errors_.HarnessClosedError(
                "harness is not open; call open() or use `async with`"
            )
        session = await self._new_session(history=history)
        try:
            return await session.run(
                prompt,
                output_type=output_type,
                retries=retries,
                timeout=timeout,
            )
        finally:
            with contextlib.suppress(Exception):
                await session.close()

    async def _new_session(
        self, history: list[messages.Message] | None = None
    ) -> _session.Session:
        session_id = await self.adapter.new_session(history=history)
        session = _session.Session(self, session_id, history=history)
        self._register(session)
        return session

    # -- conversations the harness owns ---------------------------------------

    async def sessions(self) -> list[_session.SessionInfo]:
        """List every conversation the harness has in this workspace.

        Including ones started in its own UI, by a human, before we existed.
        """
        self._require_open()
        self._require("history")
        infos = await self.adapter.list_sessions()
        running = await self.adapter.running_sessions()
        for info in infos:
            info.running = info.session_id in running
        return infos

    async def history(
        self, session_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[Any]:
        """Return a stored transcript, in full.

        Tool calls and results included, never a display summary. Same
        `ai.types.messages.Message` type as `session.messages` — one read path
        for the whole SDK.
        """
        self._require_open()
        self._require("history")
        return await self.adapter.history(
            session_id, limit=limit, offset=offset
        )

    async def resume(self, session_id: str) -> _session.Session:
        """Continue an existing conversation.

        What `claude --resume` and `codex resume` do, under the name they use.

        The transcript is loaded up front, so `session.messages` is truthful
        before you prompt. A conversation another SDK client still has open
        is refused with `SessionBusyError` — one writer per transcript, on every
        harness. A conversation a human still has open in the harness's own
        TUI holds no such claim: resuming it makes two writers, so `fork`
        it instead.
        """
        self._require_open()
        self._require("resume")
        await self.adapter.resume_session(session_id)
        session = _session.Session(self, session_id)
        session._seed(await self.adapter.history(session_id))
        self._register(session)
        return session

    async def tui(
        self,
        session_id: str | None = None,
        *,
        history: list[messages.Message] | None = None,
        name: str | None = None,
        size: tuple[int, int] | None = None,
    ) -> _session.Tui:
        """Open the harness's own interactive TUI on a pty in the workspace.

        `claude` or `codex`, for a new conversation or to continue one. Returns
        a `Tui`: the conversation's identity (`session_id`, `handle`) and the
        `pty` your terminal attaches to.

        The other way to run a harness: a human drives it, not this API.
        `sessions()` and `history()` still see the conversation, and it can
        be resumed headless later. Give it a `name` and it outlives your
        connection: detach with `tty.bridge`, reattach with
        `workspace.attach(name)` from anywhere. Drive it with
        `ai.workspaces.experimental.tty.bridge(tui.pty)`.

        `history` here means what it means on `session()`: a new conversation
        that starts with this past — how a conversation is pushed into a
        sandbox TUI, or brought home into a local one.

        On a provider-owned workspace (a sandbox) the harness's first-run
        setup is completed first, so the TUI opens on a prompt, not an
        onboarding screen. The conversation takes the session lock like any
        other writer — a new one where the CLI lets us choose its id, an
        existing one always.
        """
        self._require_open()
        if session_id is not None and history is not None:
            raise ValueError(
                "tui(): pass session_id to continue a conversation, or history "
                "to start one with a past — not both"
            )
        await self.adapter.prepare_tui()
        if history is not None:
            # `history` means the same thing it means on session(): a new
            # conversation that begins with this past. Staged in the store
            # with nothing running on it, then opened as a resume — the TUI
            # is its only writer.
            session_id = await self.adapter.stage_history(history)
            await self.adapter.claim_new_session(session_id)
        argv, env, running_as = self.adapter.tui_launch(session_id)
        if session_id is not None and history is None:
            await self.adapter.resume_session_lock(session_id)
        elif session_id is None and running_as is not None:
            await self.adapter.claim_new_session(running_as)
        pty = await self.workspace.pty(argv, env=env, name=name, size=size)
        held = session_id or running_as
        if held is not None:
            # The TUI process is now the writer: judge the lock's liveness by
            # it, not by whatever client process took the lock.
            await self.adapter.record_tui_pid(held, pty.pid)
        return _session.Tui(self, pty, held)

    async def fork(self, session_id: str) -> _session.Session:
        """Branch a conversation into a new one that inherits its past.

        The safe way to pick up a conversation a human still has open: the
        original is never written to.
        """
        self._require_open()
        self._require("fork")
        forked = await self.adapter.fork(session_id)
        session = _session.Session(self, forked)
        session._seed(await self.adapter.history(forked))
        self._register(session)
        return session

    def _require_open(self) -> None:
        if not self._open:
            raise errors_.HarnessClosedError(
                "harness is not open; call open() or use `async with`"
            )

    def _require(self, capability: str) -> None:
        if not getattr(self.capabilities, capability):
            raise errors_.UnsupportedError(self.name, capability)

    # -- approval plumbing ----------------------------------------------------

    def _register(self, session: Any) -> None:
        self._sessions[session.session_id] = session

    def _history_of(self, session_id: str) -> list[Any]:
        session = self._sessions.get(session_id)
        return session.messages if session is not None else []

    def _record_approval_error(self, detail: str) -> None:
        self._approval_errors.append(detail)

    def _take_approval_errors(self) -> list[str]:
        errors, self._approval_errors = self._approval_errors, []
        return errors

    def __repr__(self) -> str:
        state = "open" if self._open else "closed"
        return f"<Harness {self.name} ({state}) on {self.workspace.kind}>"
