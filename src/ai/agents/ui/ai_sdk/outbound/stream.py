"""Convert an internal ``ai.Message`` stream into AI SDK UI protocol parts."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable

from .....types import messages as messages_
from .. import protocol
from ._state import _StreamState


async def to_stream(
    messages: AsyncIterable[messages_.Message],
) -> AsyncGenerator[protocol.UIMessageStreamPart]:
    """Walk ``messages`` once, emitting AI SDK UI stream parts.

    Drives off ``Message.stream.events`` for incremental deltas and
    ``Message.parts`` for terminal tool input/output/approval parts.
    Re-emitted messages (same id, already seen ``is_done``) are skipped.
    """
    state = _StreamState()

    async for msg in messages:
        if msg.id in state.seen_done:
            continue

        for part in state.on_message(msg):
            yield part

        if msg.stream is not None and msg.stream.events:
            parts_by_id = {p.id: p for p in msg.parts}
            for event in msg.stream.events:
                for out in state.on_event(msg, event, parts_by_id):
                    yield out

        for part in state.on_terminal(msg):
            yield part

        if msg.is_done:
            state.seen_done.add(msg.id)

    for part in state.finish():
        yield part
