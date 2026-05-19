"""Convert internal event streams into AI SDK UI protocol parts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .....types import events as events_
from ._state import _StreamState

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable

    from .. import protocol


async def to_stream(
    events: AsyncIterable[events_.AgentEvent],
) -> AsyncGenerator[protocol.UIMessageStreamPart]:
    """Walk ``events`` once, emitting AI SDK UI stream parts.

    Streaming text/reasoning/tool-input deltas come from model events.
    Tool results come from ``ToolCallResult``.  Hook signals come from
    ``HookEvent``.
    """
    state = _StreamState()

    async for event in events:
        if isinstance(event, events_.ToolCallResult):
            for part in state.on_tool_result(event):
                yield part
        elif isinstance(event, events_.PartialToolCallResult):
            for part in state.on_partial_tool_result(event):
                yield part
        elif isinstance(event, events_.HookEvent):
            for part in state.on_hook(event):
                yield part
        else:
            for part in state.on_event(event):
                yield part

    for part in state.finish():
        yield part
