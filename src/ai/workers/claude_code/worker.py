"""`ClaudeCode`: Claude Code as a worker."""

from __future__ import annotations

from typing import Any, Literal

from ...providers import _optional
from .. import _adapter, _worker


class ClaudeCode(_worker.Worker):
    """Claude Code, through `claude-agent-sdk`. Needs the `claude-code` extra.

    `allowed_tools` limits Claude Code's own tools (`["Read", "Grep"]`, or
    `[]` for none), and `disallowed_tools` takes some away. `max_turns` caps
    the agent's steps in one turn; a turn that hits it ends with
    `finish_reason="length"`.
    """

    kind: Literal["claude-code"] = "claude-code"
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    max_turns: int | None = None

    def _make_adapter(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        approve: _adapter.Approve | None,
    ) -> _adapter.CLIAdapter:
        _optional.import_optional_sdk(
            "claude_agent_sdk",
            extra="claude-code",
            feature="ai.workers.ClaudeCode",
        )
        # Imported here: the adapter needs claude_agent_sdk at import time.
        from . import adapter  # noqa: PLC0415

        return adapter.ClaudeAdapter(
            cwd=self.cwd,
            env=dict(self.env),
            executable=self.executable or "claude",
            model=self.model,
            effort=self.effort,
            mcp_servers=dict(mcp_servers),
            approve=approve,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            max_turns=self.max_turns,
        )
