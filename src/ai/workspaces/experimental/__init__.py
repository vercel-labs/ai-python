"""Where a harness runs and its files live: this machine or a Vercel Sandbox.

Experimental: not part of the stable API, may change or be removed.

`tty` is not imported here: it needs POSIX terminals. Import it as
`from ai.workspaces.experimental import tty`.
"""

from . import errors
from ._base import (
    ExecResult,
    Process,
    Workspace,
    WorkspaceCoords,
    WorkspacePath,
    contains,
    copy,
)
from ._gateway import Gateway, vercel_ai_gateway
from ._local import Local
from ._pty import Pty, PtyInfo
from ._sandbox import VercelSandbox

__all__ = [
    "ExecResult",
    "Gateway",
    "Local",
    "Process",
    "Pty",
    "PtyInfo",
    "VercelSandbox",
    "Workspace",
    "WorkspaceCoords",
    "WorkspacePath",
    "contains",
    "copy",
    "errors",
    "vercel_ai_gateway",
]
