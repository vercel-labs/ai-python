"""Streaming: the turn is an event stream of AI SDK events, then a result.

The contract that keeps this simple: a Turn is READ-ONLY. It yields events
and holds the settled result. Control lives on the Session (see
test_control.py), so there is no dual awaitable/iterable object and no way
to consume a turn twice.
"""

from __future__ import annotations

import pytest

from ai.harnesses.experimental import Harness
from ai.types.events import (
    Event,
    ModelEvent,
    StreamEnd,
    TextDelta,
    TextEnd,
    ToolEnd,
    ToolStart,
)
from ai.types.messages import TextPart, ToolResultPart
from tests.harnesses.experimental.conftest import exact

pytestmark = pytest.mark.live


async def test_stream_yields_ai_sdk_events(any_harness: Harness) -> None:
    session = any_harness.session()
    turn = session.stream("Reply with exactly: READY")

    events = [event async for event in turn]

    assert events, "a turn always emits something"
    assert all(isinstance(e, Event) for e in events)
    assert any(isinstance(e, TextDelta) for e in events)
    assert isinstance(events[-1], StreamEnd)


async def test_text_deltas_reassemble_into_the_result_text(
    any_harness: Harness,
) -> None:
    session = any_harness.session()
    turn = session.stream("Reply with exactly: READY")

    streamed = "".join(
        [e.chunk async for e in turn if isinstance(e, TextDelta)]
    )

    assert "READY" in streamed
    assert streamed.strip() == turn.result.text.strip()


async def test_result_is_available_after_iteration(
    any_harness: Harness,
) -> None:
    session = any_harness.session()
    turn = session.stream("Reply with exactly: READY")

    async for _ in turn:
        pass

    assert turn.result.finish_reason == "stop"
    exact(turn.result.text, "READY")


async def test_reading_the_result_before_settling_is_an_error(
    any_harness: Harness,
) -> None:
    """No silent empty result: ask too early and you are told, not misled."""
    session = any_harness.session()
    turn = session.stream("Reply with exactly: READY")

    with pytest.raises(RuntimeError, match="not settled"):
        _ = turn.result

    async for _ in turn:
        pass
    assert turn.result.text


async def test_tool_calls_appear_as_tool_events(any_harness: Harness) -> None:
    session = any_harness.session()
    turn = session.stream("Read util.py and reply with the function name only.")

    starts, ends = [], []
    async for event in turn:
        if isinstance(event, ToolStart):
            starts.append(event)
        elif isinstance(event, ToolEnd):
            ends.append(event)

    assert starts, "reading a file must surface as a tool call"
    assert {e.tool_call_id for e in ends} <= {e.tool_call_id for e in starts}
    exact(turn.result.text, "add")


async def test_a_turn_cannot_be_iterated_twice(any_harness: Harness) -> None:
    session = any_harness.session()
    turn = session.stream("Reply with exactly: READY")

    first = [e async for e in turn]
    second = [e async for e in turn]

    assert first
    assert (
        second == []
    ), "a consumed stream is exhausted, never silently replayed"


async def test_every_event_carries_the_message_so_far(
    any_harness: Harness,
) -> None:
    """The AI SDK's contract: a streamed event's `.message` is the assistant
    message accumulated up to that point, so nobody has to reassemble
    deltas by hand. This SDK once broke it by discarding the hydrator's
    annotated copy — every event's `.message` was an empty placeholder
    until StreamEnd, and a consumer wanting "the complete text" had to
    build it themselves.
    """
    session = any_harness.session()
    turn = session.stream("Read util.py and reply with the function name only.")

    seen_parts = 0
    async for event in turn:
        if not isinstance(event, ModelEvent):
            # A tool result opens a new assistant segment.
            seen_parts = 0
            continue
        assert (
            event.message.id != "<unset>"
        ), f"{event.kind} carried the placeholder"
        # The message an assistant event describes is the assistant's, never
        # the tool message that happened to arrive just before it.
        assert (
            event.message.role == "assistant"
        ), f"{event.kind} landed on a {event.message.role} message"
        # Within a segment parts accumulate; nothing already streamed
        # disappears.
        assert len(event.message.parts) >= seen_parts
        seen_parts = len(event.message.parts)

        if isinstance(event, TextEnd):
            # The part this event completed is on the message, whole.
            done = next(
                p for p in event.message.parts if p.id == event.block_id
            )
            assert isinstance(done, TextPart)
            assert done.text.strip(), "TextEnd arrived with no accumulated text"
        if isinstance(event, ToolEnd):
            # The complete call rides on the event itself.
            assert event.tool_call.tool_name
            assert event.tool_call.tool_args.strip().startswith("{")

    # And the settled result is unchanged by any of this.
    exact(turn.result.text, "add")
    # History was never quietly rewritten: a tool message holds tool results
    # and nothing else, under its own id.
    tool_messages = [m for m in session.messages if m.role == "tool"]
    assert tool_messages, "the prompt is meant to make the agent use a tool"
    assistant_ids = {m.id for m in session.messages if m.role == "assistant"}
    for m in tool_messages:
        assert all(isinstance(p, ToolResultPart) for p in m.parts), m.parts
        assert m.id not in assistant_ids
