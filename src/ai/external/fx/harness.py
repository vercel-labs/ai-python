"""Fx: the provider and settings for `fx acp`."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from .. import _harness
from . import wire


class FxProvider(_harness.ExternalHarnessProvider):
    """The fx CLI. `env={"FX_HOME": ...}` isolates its settings and store."""

    name: str = "fx"

    def _make_wire(self, harness: Fx) -> wire.FxWire:
        return wire.FxWire(harness)


@dataclasses.dataclass(kw_only=True, eq=False)
class Fx(_harness.ExternalHarness):
    """Fx over `fx acp`, with full access to its built-in tools.

    `model_provider` and `effort` select the provider and reasoning effort. A
    bridge tool must answer within 5 minutes. All streamed tool results are
    short previews. Instructions are fixed at session start; resume does not
    change them. Built-in tools cannot be limited. Fx loads user settings and
    project AGENTS.md; use `FxProvider(env={"FX_HOME": ...})` to isolate
    settings. Seeding writes fx's internal session format. A new fx release can
    break it: the external harness raises instead of starting a session with a
    wrong past.
    """

    harness_class_id: ClassVar[str] = "fx"

    provider: FxProvider = dataclasses.field(default_factory=FxProvider)
    model_provider: str | None = None
    effort: str | None = None
