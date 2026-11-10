"""The errors a worker raises.

Three cases have their own class, because a caller fixes them differently.
Everything else is a `WorkerError` with a message that says what happened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import errors

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    """A session cannot take the messages given to `stream()`.

    A session holds a conversation in its CLI. It can continue it, but it
    cannot rewrite it: use `session.messages` as the history. For another
    conversation, use a worker with no `session_id`. A history must always
    end with a user message, or with the results of tool calls that the
    session made.
    """


# Phrases that mean "no usable credentials" in any CLI. Narrow on purpose: a
# 403 is authorization, not authentication. A wire adds its CLI's own phrases.
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
)


def turn_failed(
    kind: str, detail: str, hint: str, markers: Sequence[str] = ()
) -> WorkerError:
    """Name a failed turn for what the caller has to fix.

    `markers` are the CLI's own phrases for "no usable credentials".
    """
    detail = (detail or "unknown error")[:500]
    if any(m in detail.lower() for m in (*_AUTH_MARKERS, *markers)):
        return NotAuthenticatedError(kind, detail, hint)
    return WorkerError(f"{kind} turn failed: {detail}")
