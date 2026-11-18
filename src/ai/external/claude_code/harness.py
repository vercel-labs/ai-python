"""Claude Code: the provider and the settings. `wire.py` drives the CLI."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from .. import _harness
from . import wire


class ClaudeCodeProvider(_harness.ExternalHarnessProvider):
    """The Claude Code CLI."""

    name: str = "claude-code"

    def _make_wire(self, harness: ClaudeCode) -> wire.ClaudeCodeWire:
        return wire.ClaudeCodeWire(harness)


@dataclasses.dataclass(kw_only=True, eq=False)
class ClaudeCode(_harness.ExternalHarness):
    """Claude Code, through `claude-agent-sdk`. Needs the `claude-code` extra.

    `allowed_tools` limits Claude Code's own tools (`["Read", "Grep"]`, or
    `[]` for none), and `disallowed_tools` takes some away. `max_turns` caps
    the agent's steps in one turn; a turn that hits it ends with
    `finish_reason="length"`. Claude Code never asks for approval: it runs
    with `bypassPermissions`.
    """

    harness_class_id: ClassVar[str] = "claude-code"

    provider: ClaudeCodeProvider = dataclasses.field(
        default_factory=ClaudeCodeProvider
    )
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = dataclasses.field(default_factory=list)
    max_turns: int | None = None
