"""Pi: the provider and the settings. `wire.py` drives `pi --mode rpc`."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

from .. import _harness
from . import wire


class PiProvider(_harness.ExternalHarnessProvider):
    """The pi CLI."""

    name: str = "pi"

    def _make_wire(self, harness: Pi) -> wire.PiWire:
        return wire.PiWire(harness)


@dataclasses.dataclass(kw_only=True, eq=False)
class Pi(_harness.ExternalHarness):
    """Pi, over `pi --mode rpc`.

    `model_provider` picks the provider of `model_name` (`--provider`), and
    `thinking` the thinking level. `allowed_tools` limits pi's own tools
    (`["read", "grep"]`, or `[]` for none; `ai` tools stay), and
    `disallowed_tools` takes some away. `args` go to pi as they are. pi never
    asks for approval.
    """

    harness_class_id: ClassVar[str] = "pi"

    provider: PiProvider = dataclasses.field(default_factory=PiProvider)
    model_provider: str | None = None
    thinking: (
        Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
        | None
    ) = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    args: list[str] = dataclasses.field(default_factory=list)
