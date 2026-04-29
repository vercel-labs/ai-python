from __future__ import annotations

from collections.abc import AsyncGenerator

from ai.agents import events as agent_events_
from ai.agents.ui.ai_sdk import protocol, to_stream
from ai.types import events as events_
from ai.types import messages as messages_


async def _gen(
    stream_events: list[agent_events_.AgentEvent],
) -> AsyncGenerator[agent_events_.AgentEvent]:
    for event in stream_events:
        yield event


async def _collect(
    stream_events: list[agent_events_.AgentEvent],
) -> list[protocol.UIMessageStreamPart]:
    return [part async for part in to_stream(_gen(stream_events))]


def _assistant_start(
    msg_id: str = "m1",
    *,
    turn_id: str | None = "t1",
    source_label: str | None = None,
) -> agent_events_.MessageStart:
    return agent_events_.MessageStart(
        message=messages_.Message(
            id=msg_id,
            role="assistant",
            turn_id=turn_id,
            source_label=source_label,
            parts=[],
        )
    )


async def test_event_driven_text_streaming() -> None:
    text_id = "txt1"
    final = messages_.Message(
        id="m1",
        role="assistant",
        turn_id="t1",
        parts=[messages_.TextPart(id=text_id, text="hi")],
    )
    out = await _collect(
        [
            _assistant_start("m1"),
            events_.TextStart(block_id=text_id),
            events_.TextDelta(block_id=text_id, chunk="hi"),
            events_.TextEnd(block_id=text_id),
            agent_events_.MessageEnd(message=final),
        ]
    )

    assert isinstance(out[0], protocol.StartPart)
    assert out[0].message_id == "m1"
    assert isinstance(out[1], protocol.StartStepPart)
    assert isinstance(out[2], protocol.TextStartPart) and out[2].id == text_id
    assert isinstance(out[3], protocol.TextDeltaPart) and out[3].delta == "hi"
    assert isinstance(out[4], protocol.TextEndPart) and out[4].id == text_id
    assert isinstance(out[5], protocol.FinishStepPart)
    assert isinstance(out[6], protocol.FinishPart)


async def test_static_text_message_emits_text_parts() -> None:
    msg = messages_.Message(
        id="m1",
        role="assistant",
        parts=[messages_.TextPart(id="txt1", text="hello")],
    )
    out = await _collect(
        [agent_events_.MessageStart(message=msg), agent_events_.MessageEnd(message=msg)]
    )
    assert any(isinstance(part, protocol.TextDeltaPart) for part in out)


async def test_turn_id_change_emits_step_boundary() -> None:
    msg1 = messages_.Message(
        id="m1",
        role="assistant",
        turn_id="t1",
        parts=[messages_.TextPart(text="hello")],
    )
    msg2 = messages_.Message(
        id="m2",
        role="assistant",
        turn_id="t2",
        parts=[messages_.TextPart(text="world")],
    )
    out = await _collect(
        [
            agent_events_.MessageStart(message=msg1),
            agent_events_.MessageEnd(message=msg1),
            agent_events_.MessageStart(message=msg2),
            agent_events_.MessageEnd(message=msg2),
        ]
    )
    has_mid_step_boundary = any(
        isinstance(out[i], protocol.FinishStepPart)
        and i + 1 < len(out)
        and isinstance(out[i + 1], protocol.StartStepPart)
        for i in range(1, len(out) - 1)
    )
    assert has_mid_step_boundary


async def test_agent_change_emits_message_boundary() -> None:
    msg1 = messages_.Message(
        id="m1",
        role="assistant",
        source_label="a1",
        parts=[messages_.TextPart(text="from a")],
    )
    msg2 = messages_.Message(
        id="m2",
        role="assistant",
        source_label="a2",
        parts=[messages_.TextPart(text="from b")],
    )
    out = await _collect(
        [
            agent_events_.MessageStart(message=msg1),
            agent_events_.MessageEnd(message=msg1),
            agent_events_.MessageStart(message=msg2),
            agent_events_.MessageEnd(message=msg2),
        ]
    )
    has_mid_msg_boundary = any(
        isinstance(out[i], protocol.FinishPart)
        and i + 1 < len(out)
        and isinstance(out[i + 1], protocol.StartPart)
        for i in range(1, len(out) - 1)
    )
    assert has_mid_msg_boundary


async def test_tool_call_and_result_emit_terminal_parts() -> None:
    tool_call = messages_.Message(
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
    )
    tool_result = messages_.Message(
        role="tool",
        parts=[
            messages_.ToolResultPart(
                tool_call_id="tc1",
                tool_name="search",
                result={"hits": 1},
            )
        ],
    )
    out = await _collect(
        [
            agent_events_.MessageStart(message=tool_call),
            agent_events_.MessageEnd(message=tool_call),
            agent_events_.MessageStart(message=tool_result),
            agent_events_.MessageEnd(message=tool_result),
        ]
    )
    types = [type(part).__name__ for part in out]
    assert "ToolInputStartPart" in types
    assert "ToolInputAvailablePart" in types
    assert "ToolOutputAvailablePart" in types


async def test_approval_request_hook_emits_approval_part() -> None:
    tool_call = messages_.Message(
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
    )
    hook = messages_.Message(
        role="internal",
        parts=[
            messages_.HookPart(
                hook_id="approve_tc1",
                hook_type="ToolApproval",
                status="pending",
            )
        ],
    )
    out = await _collect(
        [
            agent_events_.MessageStart(message=tool_call),
            agent_events_.MessageEnd(message=tool_call),
            agent_events_.MessageStart(message=hook),
            agent_events_.MessageEnd(message=hook),
        ]
    )
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
    )
    stream_events: list[agent_events_.AgentEvent] = [
        agent_events_.MessageStart(message=msg),
        events_.TextStart(block_id="txt1"),
        events_.TextDelta(block_id="txt1", chunk="hi"),
        events_.TextEnd(block_id="txt1"),
        agent_events_.MessageEnd(message=msg),
    ]
    out = await _collect([*stream_events, *stream_events])
    text_deltas = [part for part in out if isinstance(part, protocol.TextDeltaPart)]
    assert len(text_deltas) == 1
