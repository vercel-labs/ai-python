"""Drive coding-agent CLIs you don't own: Claude Code, Codex and Pi.

Experimental: not part of the stable API, may change or be removed.

    from ai.harnesses import experimental as harnesses
    from ai.workspaces import experimental as workspaces

    workspace = workspaces.Local(".")
    async with harnesses.claude_code(workspace=workspace) as harness:
        result = await harness.run("What does this repo do?")
"""

from . import errors
from ._approval import Allow, ApprovalContext, Decision, Deny, ToolCall
from ._capabilities import ApprovalCoverage, Capabilities
from ._factories import claude_code, codex, harness_from, pi, workspace_from
from ._handle import Handle
from ._harness import Harness
from ._result import Result
from ._session import APPROVAL_HOOK_TYPE, Session, SessionInfo, Tui, Turn

__all__ = [
    "APPROVAL_HOOK_TYPE",
    "Allow",
    "ApprovalContext",
    "ApprovalCoverage",
    "Capabilities",
    "Decision",
    "Deny",
    "Handle",
    "Harness",
    "Result",
    "Session",
    "SessionInfo",
    "ToolCall",
    "Tui",
    "Turn",
    "claude_code",
    "codex",
    "errors",
    "harness_from",
    "pi",
    "workspace_from",
]
