"""Convert an internal ``ai.Event`` stream into AI SDK UI protocol parts."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable

from ....events import AgentEvent, MessageEnd, MessageStart
from .. import protocol
from ._state import _StreamState


async def to_stream(
    events: AsyncIterable[AgentEvent],
) -> AsyncGenerator[protocol.UIMessageStreamPart]:
    """Walk ``events`` once, emitting AI SDK UI stream parts.

    Streaming text/reasoning/tool-input deltas come from public events.
    Terminal tool results, approvals, and files come from
    ``MessageEnd.message``.
    """
    state = _StreamState()

    async for event in events:
        if isinstance(event, MessageStart):
            for part in state.on_message_start(event.message):
                yield part
        elif isinstance(event, MessageEnd):
            for part in state.on_terminal(event.message):
                yield part
        else:
            for part in state.on_event(event):
                yield part

    for part in state.finish():
        yield part
