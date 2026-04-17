from __future__ import annotations

from collections.abc import AsyncGenerator

from ai.agents.ui.ai_sdk import protocol, to_stream
from ai.types import messages as messages_


async def _gen(
    msgs: list[messages_.Message],
) -> AsyncGenerator[messages_.Message]:
    for m in msgs:
        yield m


async def _collect(
    msgs: list[messages_.Message],
) -> list[protocol.UIMessageStreamPart]:
    return [p async for p in to_stream(_gen(msgs))]


def _text_stream_message(
    msg_id: str,
    turn_id: str | None,
    text_id: str,
    chunk: str,
    *,
    is_done: bool,
    full_text: str | None = None,
) -> messages_.Message:
    events: list[messages_.StreamEvent]
    if is_done:
        events = [messages_.PartClosed(part_id=text_id)]
    else:
        events = [messages_.PartDelta(part_id=text_id, chunk=chunk)]
    return messages_.Message(
        id=msg_id,
        role="assistant",
        turn_id=turn_id,
        parts=[messages_.TextPart(id=text_id, text=full_text or chunk)],
        stream=messages_.MessageStreamState(events=events, is_done=is_done),
    )


async def test_event_driven_text_streaming() -> None:
    text_id = "txt1"
    msgs = [
        # Initial: PartOpened
        messages_.Message(
            id="m1",
            role="assistant",
            turn_id="t1",
            parts=[messages_.TextPart(id=text_id, text="")],
            stream=messages_.MessageStreamState(
                events=[messages_.PartOpened(part_id=text_id)],
                is_done=False,
            ),
        ),
        # Delta: "hi"
        messages_.Message(
            id="m1",
            role="assistant",
            turn_id="t1",
            parts=[messages_.TextPart(id=text_id, text="hi")],
            stream=messages_.MessageStreamState(
                events=[messages_.PartDelta(part_id=text_id, chunk="hi")],
                is_done=False,
            ),
        ),
        # Closed
        messages_.Message(
            id="m1",
            role="assistant",
            turn_id="t1",
            parts=[messages_.TextPart(id=text_id, text="hi")],
            stream=messages_.MessageStreamState(
                events=[messages_.PartClosed(part_id=text_id)],
                is_done=True,
            ),
        ),
    ]
    out = await _collect(msgs)
    # expect: Start, StartStep, TextStart, TextDelta, TextEnd, FinishStep, Finish
    assert isinstance(out[0], protocol.StartPart)
    assert out[0].message_id == "m1"
    assert isinstance(out[1], protocol.StartStepPart)
    assert isinstance(out[2], protocol.TextStartPart) and out[2].id == text_id
    assert isinstance(out[3], protocol.TextDeltaPart) and out[3].delta == "hi"
    assert isinstance(out[4], protocol.TextEndPart) and out[4].id == text_id
    assert isinstance(out[5], protocol.FinishStepPart)
    assert isinstance(out[6], protocol.FinishPart)


async def test_turn_id_change_emits_step_boundary() -> None:
    msgs = [
        messages_.Message(
            id="m1",
            role="assistant",
            turn_id="t1",
            parts=[messages_.TextPart(text="hello")],
            stream=messages_.MessageStreamState(events=[], is_done=True),
        ),
        messages_.Message(
            id="m2",
            role="assistant",
            turn_id="t2",  # different turn → step boundary
            parts=[messages_.TextPart(text="world")],
            stream=messages_.MessageStreamState(events=[], is_done=True),
        ),
    ]
    out = await _collect(msgs)
    # Look for FinishStep followed by StartStep between messages.
    has_mid_step_boundary = any(
        isinstance(out[i], protocol.FinishStepPart)
        and i + 1 < len(out)
        and isinstance(out[i + 1], protocol.StartStepPart)
        for i in range(1, len(out) - 1)
    )
    assert has_mid_step_boundary


async def test_agent_change_emits_message_boundary() -> None:
    msgs = [
        messages_.Message(
            id="m1",
            role="assistant",
            agent="a1",
            parts=[messages_.TextPart(text="from a")],
            stream=messages_.MessageStreamState(events=[], is_done=True),
        ),
        messages_.Message(
            id="m2",
            role="assistant",
            agent="a2",  # different agent → FinishPart + StartPart
            parts=[messages_.TextPart(text="from b")],
            stream=messages_.MessageStreamState(events=[], is_done=True),
        ),
    ]
    out = await _collect(msgs)
    # There should be a FinishPart+StartPart pair mid-stream.
    has_mid_msg_boundary = any(
        isinstance(out[i], protocol.FinishPart)
        and i + 1 < len(out)
        and isinstance(out[i + 1], protocol.StartPart)
        for i in range(1, len(out) - 1)
    )
    assert has_mid_msg_boundary


async def test_tool_call_and_result_emit_terminal_parts() -> None:
    msgs = [
        messages_.Message(
            id="m1",
            role="assistant",
            turn_id="t1",
            parts=[
                messages_.ToolCallPart(
                    id="tc1",
                    tool_call_id="tc1",
                    tool_name="search",
                    tool_args='{"q":"x"}',
                )
            ],
            stream=messages_.MessageStreamState(events=[], is_done=True),
        ),
        messages_.Message(
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    tool_call_id="tc1",
                    tool_name="search",
                    result={"hits": 1},
                )
            ],
        ),
    ]
    out = await _collect(msgs)
    types = [type(p).__name__ for p in out]
    assert "ToolInputStartPart" in types
    assert "ToolInputAvailablePart" in types
    assert "ToolOutputAvailablePart" in types


async def test_approval_request_hook_emits_approval_part() -> None:
    msgs = [
        messages_.Message(
            id="m1",
            role="assistant",
            turn_id="t1",
            parts=[
                messages_.ToolCallPart(
                    id="tc1",
                    tool_call_id="tc1",
                    tool_name="delete",
                    tool_args="{}",
                )
            ],
            stream=messages_.MessageStreamState(events=[], is_done=True),
        ),
        messages_.Message(
            role="internal",
            parts=[
                messages_.HookPart(
                    hook_id="approve_tc1",
                    hook_type="ToolApproval",
                    status="pending",
                )
            ],
        ),
    ]
    out = await _collect(msgs)
    approval_parts = [p for p in out if isinstance(p, protocol.ToolApprovalRequestPart)]
    assert len(approval_parts) == 1
    assert approval_parts[0].tool_call_id == "tc1"
    assert approval_parts[0].approval_id == "approve_tc1"


async def test_dedup_on_reemitted_message_id() -> None:
    msg = messages_.Message(
        id="m1",
        role="assistant",
        turn_id="t1",
        parts=[messages_.TextPart(id="txt1", text="hi")],
        stream=messages_.MessageStreamState(
            events=[
                messages_.PartOpened(part_id="txt1"),
                messages_.PartDelta(part_id="txt1", chunk="hi"),
                messages_.PartClosed(part_id="txt1"),
            ],
            is_done=True,
        ),
    )
    out = await _collect([msg, msg])  # re-emit the same done message
    text_deltas = [p for p in out if isinstance(p, protocol.TextDeltaPart)]
    # only the first emission should fire a TextDelta
    assert len(text_deltas) == 1
