"""The workers: Claude Code, Codex and Pi.

Each adds its CLI's own options to `Worker` and builds its adapter.
"""

from __future__ import annotations

from typing import Any, Literal

import pydantic

from ..providers import _optional
from . import _bridge, _worker
from ._adapters import base, codex, pi


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
        approve: base.Approve | None,
    ) -> base.Adapter:
        _optional.import_optional_sdk(
            "claude_agent_sdk",
            extra="claude-code",
            feature="ai.workers.ClaudeCode",
        )
        # Imported here: the adapter needs claude_agent_sdk at import time.
        from ._adapters import claude  # noqa: PLC0415

        return claude.ClaudeAdapter(
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


class Codex(_worker.Worker):
    """Codex, over `codex app-server`.

    `sandbox` is Codex's own sandbox mode: `"read-only"`,
    `"workspace-write"` or `"danger-full-access"` (the default). `config`
    takes Codex `-c` overrides.
    """

    kind: Literal["codex"] = "codex"
    sandbox: (
        Literal["read-only", "workspace-write", "danger-full-access"] | None
    ) = None
    config: dict[str, Any] = pydantic.Field(default_factory=dict)

    def _make_adapter(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        approve: base.Approve | None,
    ) -> base.Adapter:
        config = dict(self.config)
        if self.effort is not None:
            config.setdefault(
                "model_reasoning_effort",
                "xhigh" if self.effort == "max" else self.effort,
            )
        if _bridge.SERVER_NAME in mcp_servers:
            # Codex's default wait for an MCP tool (60s) is too short for a
            # tool an `ai.Agent` runs, maybe after an approval.
            config.setdefault(
                f"mcp_servers.{_bridge.SERVER_NAME}.tool_timeout_sec",
                24 * 3600,
            )
        return codex.CodexAdapter(
            cwd=self.cwd,
            env=dict(self.env),
            executable=self.executable or "codex",
            model=self.model,
            mcp_servers=dict(mcp_servers),
            approve=approve,
            sandbox=self.sandbox,
            config=config,
        )


class Pi(_worker.Worker):
    """Pi, over `pi --mode rpc`.

    `allowed_tools` limits pi's own tools (`["read", "grep"]`, or `[]` for
    none), and `disallowed_tools` takes some away.
    """

    kind: Literal["pi"] = "pi"
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    def _make_adapter(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        approve: base.Approve | None,
    ) -> base.Adapter:
        return pi.PiAdapter(
            cwd=self.cwd,
            env=dict(self.env),
            executable=self.executable or "pi",
            model=self.model,
            effort=self.effort,
            mcp_servers=dict(mcp_servers),
            approve=approve,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
        )
