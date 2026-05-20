"""Convert internal event streams into AI SDK UI protocol events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .....types import events as events_
from ._state import _StreamState

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable

    from .. import ui_events


async def to_stream(
    events: AsyncIterable[events_.AgentEvent],
) -> AsyncGenerator[ui_events.UIMessageStreamEvent]:
    """Walk ``events`` once, emitting AI SDK UI stream events.

    Streaming text/reasoning/tool-input deltas come from model events.
    Tool results come from ``ToolCallResult``.  Hook signals come from
    ``HookEvent``.
    """
    state = _StreamState()

    async for event in events:
        if isinstance(event, events_.ToolCallResult):
            for ui_event in state.on_tool_result(event):
                yield ui_event
        elif isinstance(event, events_.PartialToolCallResult):
            for ui_event in state.on_partial_tool_result(event):
                yield ui_event
        elif isinstance(event, events_.HookEvent):
            for ui_event in state.on_hook(event):
                yield ui_event
        else:
            for ui_event in state.on_event(event):
                yield ui_event

    for ui_event in state.finish():
        yield ui_event
