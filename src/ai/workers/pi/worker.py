"""Pi: the settings. `wire.py` drives `pi --mode rpc`."""

from __future__ import annotations

from typing import Literal

import pydantic

from .. import _worker
from . import wire


class Pi(_worker.Worker):
    """Pi, over `pi --mode rpc`.

    `provider` picks the provider of `model_name` (`--provider`), and
    `thinking` the thinking level. `allowed_tools` limits pi's own tools
    (`["read", "grep"]`, or `[]` for none; `ai` tools stay), and
    `disallowed_tools` takes some away. `args` go to pi as they are. pi never
    asks for approval.
    """

    worker_class_id: Literal["pi"] = "pi"
    provider: str | None = None
    thinking: (
        Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
        | None
    ) = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    args: list[str] = pydantic.Field(default_factory=list)

    def _make_wire(self) -> wire.PiWire:
        return wire.PiWire(self)
