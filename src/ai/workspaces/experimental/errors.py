"""Workspace errors.

Experimental: not part of the stable API, may change or be removed.
"""

from __future__ import annotations

from ... import errors as ai_errors


class WorkspaceError(ai_errors.AIError):
    """The workspace could not do what was asked of it."""


class WorkspaceGoneError(WorkspaceError):
    """The workspace no longer exists.

    The provider says it was stopped or was never there, as opposed to one
    that could not be reached right now.

    The difference decides what a caller may forget. A network failure is a
    WorkspaceError: the sandbox may be running, and whoever recorded it must
    keep that record or lose the only way back to it.
    """


class NotAuthenticatedError(WorkspaceError):
    """The harness has no usable credentials WHERE IT IS RUNNING.

    Its own report is an upstream HTTP status, which tells you nothing you
    can act on. The distinction that matters is location: your terminal
    login lives on your machine, and a sandbox is a different machine that
    has never seen it.
    """

    def __init__(self, harness: str, detail: str, hint: str = "") -> None:
        message = f"{harness} is not authenticated in this workspace: {detail}"
        super().__init__(f"{message}\n{hint}" if hint else message)
        self.harness = harness
        self.detail = detail
        self.hint = hint
