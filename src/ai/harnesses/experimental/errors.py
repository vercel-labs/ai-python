"""Harness errors.

The taxonomy is small on purpose: every entry answers a question the caller
can act on.

Experimental: not part of the stable API, may change or be removed.
"""

from __future__ import annotations

from ... import errors as ai_errors
from ...workspaces.experimental import errors as workspace_errors

NotAuthenticatedError = workspace_errors.NotAuthenticatedError


class HarnessError(ai_errors.AIError):
    """Base for everything the harnesses raise deliberately."""


class UnsupportedError(HarnessError):
    """The harness cannot do this, and saying so is the contract.

    Never a silent no-op: a capability that is quietly ignored looks exactly
    like one that works, which is the single most expensive lie this SDK can
    tell.
    """

    def __init__(self, harness: str, capability: str, detail: str = "") -> None:
        message = f"{harness} does not support {capability}"
        super().__init__(f"{message}: {detail}" if detail else message)
        self.harness = harness
        self.capability = capability


class HarnessClosedError(HarnessError):
    """The harness was closed; open a new one."""


class ExecutableMissingError(HarnessError):
    """The harness binary is not installed where the workspace can run it."""

    def __init__(
        self, kind: str, executable: str, install: str | None = None
    ) -> None:
        hint = f" — install it with: {install}" if install else ""
        super().__init__(
            f"{kind} needs {executable!r}, which is not runnable{hint}"
        )
        self.kind = kind
        self.executable = executable
        self.install = install


class SessionBusyError(HarnessError):
    """Another client holds this conversation open.

    One writer per conversation, on every harness: codex enforces it in
    app-server; the SDK enforces it for the others with a lock kept in the
    workspace. `fork(session_id)` continues from the same history in a
    conversation of your own; `close()` on the holder ends its claim.
    """

    def __init__(
        self, session_id: str, holder_pid: int | None = None, detail: str = ""
    ) -> None:
        self.session_id = session_id
        self.holder_pid = holder_pid
        who = f"process {holder_pid}" if holder_pid else "another client"
        message = (
            f"session {session_id} is held by {who}: fork(session_id) to "
            "continue " + "from its history, or close() the holder"
        )
        if detail:
            message += f" ({detail})"
        super().__init__(message)


class ResumeFailedError(HarnessError):
    """The named conversation could not be reopened."""

    def __init__(self, session_id: str, detail: str) -> None:
        super().__init__(f"cannot resume session {session_id}: {detail}")
        self.session_id = session_id


class AgentCrashedError(HarnessError):
    """The harness process died while we were driving it.

    Carries whatever evidence the harness library gave us, because "the turn
    failed" without a reason is the least actionable error there is.
    """

    def __init__(
        self, harness: str, detail: str, exit_code: int | None = None
    ) -> None:
        code = f" (exit {exit_code})" if exit_code is not None else ""
        super().__init__(f"{harness} crashed{code}: {detail}")
        self.harness = harness
        self.detail = detail
        self.exit_code = exit_code


class TurnFailedError(HarnessError):
    """The harness reported the turn itself as failed.

    Distinct from a crash (the process is fine) and from a refusal (the
    model declining is a normal settlement, not an error).
    """

    def __init__(self, harness: str, detail: str) -> None:
        super().__init__(f"{harness} turn failed: {detail}")
        self.harness = harness
        self.detail = detail


# Phrases that mean "no usable credentials", collected from what the two
# harnesses actually print. Deliberately narrow: a 403 is authorization,
# not authentication, and mislabelling a permissions problem as a login
# problem sends someone to fix the wrong thing.
_AUTH_MARKERS = (
    "401",
    "unauthorized",
    "missing bearer",
    "not logged in",
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "no api key",
    "please run /login",
    "please run `codex login`",
)


def _looks_unauthenticated(detail: str) -> bool:
    """Whether a harness's failure text is a credentials problem."""
    lowered = (detail or "").lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)
