"""The adapter seam: what a harness must provide, and nothing more.

An adapter yields a stream of AI SDK events that ENDS with a StreamEnd.
That single shape is why there is no dual awaitable/iterable turn object,
no subscriber queues, and no sentinels: `run()` drains the stream,
`stream()` forwards it, and both are the same code path.

Adapters never spawn processes themselves. They ask the workspace, which is
the only reason the same adapter drives a CLI here or inside a microVM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from .. import _approval, _capabilities

if TYPE_CHECKING:
    from ....types import events, messages
    from ....workspaces.experimental import _base

ApprovalHook = Callable[
    [_approval.ApprovalContext], Awaitable[_approval.Decision]
]


class Adapter(Protocol):
    """One harness, driven natively."""

    kind: str
    capabilities: _capabilities.Capabilities

    @property
    def version(self) -> str | None:
        """Whatever the CLI reported when it was probed."""
        ...

    async def start(
        self, workspace: _base.Workspace, options: dict[str, Any]
    ) -> None:
        """Probe the executable and prepare to work.

        Raises ExecutableMissingError when the binary is not runnable in the
        workspace.
        """
        ...

    async def close(self) -> None: ...

    async def detach(self) -> None:
        """Release without terminating.

        For workspaces that can outlive the client. Defaults to close() for
        harnesses that cannot survive.
        """
        await self.close()

    @property
    def process_ids(self) -> list[int]:
        """OS pids this adapter owns, for teardown assertions.

        Empty when the processes live somewhere this machine cannot signal.
        """
        ...

    async def new_session(
        self, history: list[messages.Message] | None = None
    ) -> str:
        """Create a conversation.

        With `history`, the harness stores those messages natively first, so the
        agent begins with that context.
        """
        ...

    async def stage_history(self, history: list[messages.Message]) -> str:
        """Store a conversation with this past, and return its id.

        NOTHING runs on it. What `session(history=)` does before it connects,
        and what `tui(history=)` needs alone.
        """
        ...

    async def resume_session(self, session_id: str) -> None: ...

    async def close_session(self, session_id: str) -> None: ...

    def turn(
        self, session_id: str, prompt: str
    ) -> AsyncIterator[events.AgentEvent]:
        """Drive one turn, yielding AI SDK events and ending with StreamEnd."""
        ...

    async def prepare_tui(self) -> None:
        """Make a PROVIDER-OWNED workspace ready for the interactive CLI.

        First-run onboarding done, provider config on disk, so a fresh sandbox's
        TUI opens on a prompt instead of a setup screen. Never touches a user's
        own machine. Default: nothing to do.
        """

    def tui_launch(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        """Return the interactive command, its env, and its session id.

        The command is for a new conversation or an existing one; the env is
        what it needs (credentials, TERM); the session id is minted here for a
        new conversation when the CLI lets us choose it (claude), None when the
        CLI mints its own (codex).
        """
        raise NotImplementedError

    async def record_tui_pid(self, session_id: str, pid: int | None) -> None:
        """Point the conversation's lock at the TUI process.

        Liveness is then judged by the program that actually holds it.
        """

    async def claim_new_session(self, session_id: str) -> None:
        """Lock a new TUI conversation whose id we minted.

        It then shows as running and refuses a second writer.
        """

    async def release_tui_lock(self, session_id: str) -> None:
        """Release the claim a TUI conversation held, once its TUI ended."""

    async def resume_session_lock(self, session_id: str) -> None:
        """Claim an EXISTING conversation before opening its TUI on it.

        One writer, whichever way it is driven.
        """

    async def running_sessions(self) -> dict[str, int | None]:
        """Conversations a process has open, and its pid when known.

        A client of this SDK (see session_lock), or the CLI however it was
        started.
        """
        return {}

    async def list_sessions(self) -> list[Any]:
        """Conversations the harness has in this workspace."""
        ...

    # Set by the harness when it opens, so the adapter can call them.
    approve: ApprovalHook | None
    approval_timeout: float
    on_approval_error: Any

    live_history: Any
    """Set by the harness: the live transcript of a session, for the approval
    hook. Distinct from `history()`, which reads the STORED record."""

    async def history(
        self, session_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[Any]:
        """Return a stored transcript as AI SDK messages.

        The SAME type as `session.messages`, and always the full record.
        """
        ...

    async def fork(self, session_id: str) -> str:
        """Branch a conversation; return the new session id."""
        ...

    async def steer(self, session_id: str, text: str) -> None:
        """Inject a message into the RUNNING turn.

        As soon as the harness will take it. Raises UnsupportedError where there
        is no such channel.
        """
        ...

    async def stop(self, session_id: str) -> None:
        """Graceful interrupt. The harness keeps its transcript."""
        ...
