"""The adapter seam: what a harness must provide, and nothing more.

An adapter yields a stream of AI SDK events that ENDS with a StreamEnd.
That single shape is why there is no dual awaitable/iterable turn object,
no subscriber queues, and no sentinels: `run()` drains the stream,
`stream()` forwards it, and both are the same code path.

Adapters never spawn processes themselves. They ask the workspace, which is
the only reason the same adapter drives a CLI here or inside a microVM.
"""

from __future__ import annotations

import contextlib
import posixpath
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from .. import _approval, _capabilities, _cli_processes, _input

if TYPE_CHECKING:
    from ....types import events, messages
    from ....workspaces.experimental import _base

ApprovalHook = Callable[
    [_approval.ApprovalContext], Awaitable[_approval.Decision]
]


async def provision(
    workspace: _base.Workspace, executable: str, package: str
) -> _base.ExecResult:
    """Probe the CLI; on a provider-owned workspace, make it the latest.

    Returns the final `<executable> --version` result. On your own machine
    this only probes: the SDK never installs software there. In a sandbox it
    installs the CLI when it is missing, and upgrades it when npm has a
    newer release — unless a copy of the CLI is running there (an agent
    left running for a later reconnect), which an upgrade must not pull the
    files out from under.
    """
    probe = await workspace.exec([executable, "--version"], timeout=60)
    if workspace.owner != "provider":
        return probe
    # When npm was last asked, in the VM: asking costs ~0.6-1s, and every
    # reconnect asked (measured: most of a listing's time). The answer
    # holds for CHECK_EVERY.
    marker = f"{await workspace.home()}/.ai-python/checked/" + package.replace(
        "/", "_"
    )
    if probe.exit_code == 0:
        with contextlib.suppress(FileNotFoundError, ValueError):
            if time.time() - float(await workspace.read_text(marker)) < (
                CHECK_EVERY
            ):
                return probe
        latest = await workspace.exec(
            ["npm", "view", package, "version"], timeout=120
        )
        newest = latest.stdout.strip() if latest.exit_code == 0 else ""
        if not newest or newest in probe.stdout.split():
            if newest:
                await workspace.write_text(marker, str(time.time()))
            return probe
        for process in (await _cli_processes.processes(workspace)).values():
            if any(
                posixpath.basename(arg) == executable or package in arg
                for arg in process.argv[:2]
            ):
                return probe
    for attempt in range(2):
        try:
            await workspace.exec(
                ["npm", "install", "-g", f"{package}@latest"], timeout=600
            )
        except Exception:
            # The answer was lost, not necessarily the install (measured: a
            # sandbox's response stream cut off mid-install under load). It
            # is idempotent: look, and try once more if it is not there.
            if attempt == 1:
                raise
        probe = await workspace.exec([executable, "--version"], timeout=60)
        if probe.exit_code == 0:
            break
    if probe.exit_code == 0:
        await workspace.write_text(marker, str(time.time()))
    return probe


CHECK_EVERY = 12 * 3600.0
"""Seconds a sandbox's "this CLI is the latest" answer is trusted."""


class Adapter(Protocol):
    """One harness, driven natively."""

    kind: str
    capabilities: _capabilities.Capabilities
    verified_version: str | None
    """The CLI version the live suite last passed against; an older CLI
    gets a warning when the harness opens."""

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
        self, session_id: str, prompt: list[_input.Part]
    ) -> AsyncIterator[events.AgentEvent]:
        """Drive one turn, yielding AI SDK events and ending with StreamEnd.

        `prompt` holds text and the files the harness's capabilities admit;
        the session has already checked them (`_input.check`).
        """
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
    """Set by the harness: `on_approval_error(session_id, detail)` records a
    hook that raised or timed out, for that session's turn to report."""

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
