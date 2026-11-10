"""Pi: the provider and the settings. `wire.py` drives `pi --mode rpc`."""

from __future__ import annotations

from typing import Literal

import pydantic

from .. import _worker
from . import wire


class PiProvider(_worker.WorkerProvider):
    """The pi CLI."""

    name: str = "pi"

    def _make_wire(self, worker: Pi) -> wire.PiWire:
        return wire.PiWire(worker)


class Pi(_worker.Worker):
    """Pi, over `pi --mode rpc`.

    `model_provider` picks the provider of `model_name` (`--provider`), and
    `thinking` the thinking level. `allowed_tools` limits pi's own tools
    (`["read", "grep"]`, or `[]` for none; `ai` tools stay), and
    `disallowed_tools` takes some away. `args` go to pi as they are. pi never
    asks for approval.
    """

    worker_class_id: Literal["pi"] = "pi"
    provider: PiProvider = pydantic.Field(default_factory=PiProvider)
    model_provider: str | None = None
    thinking: (
        Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
        | None
    ) = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    args: list[str] = pydantic.Field(default_factory=list)
