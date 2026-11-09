"""Codex: the settings. `wire.py` drives `codex app-server`."""

from __future__ import annotations

from typing import Any, Literal

import pydantic

from .. import _worker
from . import wire


class Codex(_worker.Worker):
    """Codex, over `codex app-server`.

    `sandbox` is Codex's own sandbox mode: `"read-only"`,
    `"workspace-write"` or `"danger-full-access"` (the default). Codex never
    asks for approval. `config` takes Codex `-c` overrides, for example
    `{"model_reasoning_effort": "high"}`.
    """

    worker_class_id: Literal["codex"] = "codex"
    sandbox: (
        Literal["read-only", "workspace-write", "danger-full-access"] | None
    ) = None
    config: dict[str, Any] = pydantic.Field(default_factory=dict)

    def _make_wire(self) -> wire.CodexWire:
        return wire.CodexWire(self)
