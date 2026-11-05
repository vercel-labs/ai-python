"""The events the harness loop yields on top of ``ai.events.AgentEvent``."""

from __future__ import annotations

from typing import Literal

from ..types import events
from ..types import messages as messages_
from ..types import usage as usage_


class Steered(events.BaseEvent):
    """Steering messages were added to the history."""

    messages: tuple[messages_.Message, ...]
    kind: Literal["harness_steered"] = "harness_steered"


class Compacted(events.BaseEvent):
    """Old history was replaced by a summary."""

    summary: messages_.Message
    removed: int
    usage: usage_.Usage | None
    kind: Literal["harness_compacted"] = "harness_compacted"


class LimitReached(events.BaseEvent):
    """The run stopped at ``MAX_TURNS``, ``MAX_TOKENS`` or ``MAX_USD``.

    ``usage`` and ``usd`` are the budget's totals: this run and the runs
    that share its budget (subagents).
    """

    limit: Literal["turns", "tokens", "usd"]
    usage: usage_.Usage
    usd: float
    kind: Literal["harness_limit_reached"] = "harness_limit_reached"


type HarnessEvent = events.AgentEvent | Steered | Compacted | LimitReached
