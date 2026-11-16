"""Claude Code: the provider and the settings. `wire.py` drives the CLI."""

from __future__ import annotations

from typing import Literal

import pydantic

from .. import _harness
from . import wire


class ClaudeCodeProvider(_harness.ExternalHarnessProvider):
    """The Claude Code CLI."""

    name: str = "claude-code"

    def _make_wire(self, harness: ClaudeCode) -> wire.ClaudeCodeWire:
        return wire.ClaudeCodeWire(harness)


class ClaudeCode(_harness.ExternalHarness):
    """Claude Code, through `claude-agent-sdk`. Needs the `claude-code` extra.

    `allowed_tools` limits Claude Code's own tools (`["Read", "Grep"]`, or
    `[]` for none), and `disallowed_tools` takes some away. `max_turns` caps
    the agent's steps in one turn; a turn that hits it ends with
    `finish_reason="length"`. Claude Code never asks for approval: it runs
    with `bypassPermissions`.
    """

    harness_class_id: Literal["claude-code"] = "claude-code"
    provider: ClaudeCodeProvider = pydantic.Field(
        default_factory=ClaudeCodeProvider
    )
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = pydantic.Field(default_factory=list)
    max_turns: int | None = None
