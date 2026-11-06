"""`Codex`: Codex as a worker."""

from __future__ import annotations

from typing import Any, Literal

import pydantic

from .. import _adapter, _bridge, _worker
from . import adapter


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
        approve: _adapter.Approve | None,
    ) -> _adapter.CLIAdapter:
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
        return adapter.CodexAdapter(
            cwd=self.cwd,
            env=dict(self.env),
            executable=self.executable or "codex",
            model=self.model,
            mcp_servers=dict(mcp_servers),
            approve=approve,
            sandbox=self.sandbox,
            config=config,
        )
