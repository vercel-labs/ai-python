"""Codex: the provider and the settings. `wire.py` drives `codex app-server`."""

from __future__ import annotations

from typing import Any, Literal

import pydantic

from .. import _harness
from . import wire


class CodexProvider(_harness.ExternalHarnessProvider):
    """The Codex CLI."""

    name: str = "codex"

    def _make_wire(self, harness: Codex) -> wire.CodexWire:
        return wire.CodexWire(harness)


class Codex(_harness.ExternalHarness):
    """Codex, over `codex app-server`.

    `sandbox` is Codex's own sandbox mode: `"read-only"`,
    `"workspace-write"` or `"danger-full-access"` (the default). Codex never
    asks for approval. `config` takes Codex `-c` overrides, for example
    `{"model_reasoning_effort": "high"}`.
    """

    harness_class_id: Literal["codex"] = "codex"
    provider: CodexProvider = pydantic.Field(default_factory=CodexProvider)
    sandbox: (
        Literal["read-only", "workspace-write", "danger-full-access"] | None
    ) = None
    config: dict[str, Any] = pydantic.Field(default_factory=dict)
