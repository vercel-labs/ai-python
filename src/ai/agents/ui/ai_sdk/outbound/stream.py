"""Convert an internal ``ai.Event`` stream into AI SDK UI protocol parts."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable

from ....events import AgentEvent, HookEvent, PartialToolCallResult, ToolCallResult
from .. import protocol
from ._state import _StreamState


async def to_stream(
    events: AsyncIterable[AgentEvent],
) -> AsyncGenerator[protocol.UIMessageStreamPart]:
    """Walk ``events`` once, emitting AI SDK UI stream parts.

    Streaming text/reasoning/tool-input deltas come from model events.
    Tool results come from ``ToolCallResult``.  Hook signals come from
    ``HookEvent``.
    """
    state = _StreamState()

    async for event in events:
        if isinstance(event, ToolCallResult):
            for part in state.on_tool_result(event):
                yield part
        elif isinstance(event, PartialToolCallResult):
            for part in state.on_partial_tool_result(event):
                yield part
        elif isinstance(event, HookEvent):
            for part in state.on_hook(event):
                yield part
        else:
            for part in state.on_event(event):
                yield part

    for part in state.finish():
        yield part
