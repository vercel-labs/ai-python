"""`Pi`: pi as a worker."""

from __future__ import annotations

from typing import Any, Literal

from .. import _adapter, _worker
from . import adapter


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
        approve: _adapter.Approve | None,
    ) -> _adapter.CLIAdapter:
        return adapter.PiAdapter(
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
