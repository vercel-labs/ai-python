"""The errors a worker raises.

Three cases have their own class, because a caller fixes them differently.
Everything else is a `WorkerError` with a message that says what happened.
"""

from __future__ import annotations

from .. import errors


class WorkerError(errors.AIError):
    """A worker or its CLI failed."""


class UnsupportedError(WorkerError):
    """The CLI does not support the requested operation."""


class ExecutableMissingError(WorkerError):
    """The CLI is not installed, or `executable=` does not run."""

    def __init__(self, kind: str, executable: str, install: str) -> None:
        super().__init__(
            f"{kind} needs {executable!r}, which does not run here. "
            f"Install it with: {install}"
        )
        self.kind = kind
        self.executable = executable


class NotAuthenticatedError(WorkerError):
    """The CLI has no usable credentials."""

    def __init__(self, kind: str, detail: str, hint: str) -> None:
        super().__init__(f"{kind} is not authenticated: {detail}\n{hint}")
        self.kind = kind
        self.detail = detail


class TranscriptMismatchError(WorkerError):
    """The messages given to `stream()` do not continue this conversation.

    A worker holds a conversation in its CLI. It can continue it, or begin a
    new one from a history, but it cannot rewrite it. Use a new worker.
    """


# Phrases that mean "no usable credentials", collected from what the CLIs
# print. Narrow on purpose: a 403 is authorization, not authentication.
_AUTH_MARKERS = (
    "401",
    "unauthorized",
    "missing bearer",
    "not logged in",
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "failed to authenticate",
    "no api key",
    "please run /login",
    "please run `codex login`",
)


def turn_failed(kind: str, detail: str, hint: str) -> WorkerError:
    """Name a failed turn for what the caller has to fix."""
    detail = (detail or "unknown error")[:500]
    if any(marker in detail.lower() for marker in _AUTH_MARKERS):
        return NotAuthenticatedError(kind, detail, hint)
    return WorkerError(f"{kind} turn failed: {detail}")
