"""Agent-layer event types.

The model layer emits ``StreamStart`` / ``StreamEnd`` plus block-level
deltas. The agent layer wraps those with ``MessageStart`` / ``MessageEnd``
boundaries that delimit complete messages — assistant turns produced by
the model, plus synthetic user / tool / hook messages injected into the
runtime queue.

These types live here (rather than in ``ai.types.events``) because they
are an agent-runtime concern, not part of the public model-streaming
event vocabulary.
"""

from __future__ import annotations

from typing import Literal

import pydantic

from .. import types


class MessageStart(pydantic.BaseModel):
    message: types.Message | None = None

    kind: Literal["message_start"] = "message_start"


class MessageEnd(pydantic.BaseModel):
    message: types.Message
    usage: types.Usage | None = None

    kind: Literal["message_end"] = "message_end"


# Widened event alias used inside agents/.  Not part of ``types.Event``'s
# discriminated union — these wrappers do not flow through model adapters.
AgentEvent = types.Event | MessageStart | MessageEnd


__all__ = [
    "AgentEvent",
    "MessageEnd",
    "MessageStart",
]
