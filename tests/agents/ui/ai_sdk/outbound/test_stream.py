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


async def test_event_driven_text_streaming() -> None:
    """Streaming text events lazily open a UI message."""
    text_id = "txt1"
    out = await _collect(
        [
            events_.TextStart(block_id=text_id),
            events_.TextDelta(block_id=text_id, chunk="hi"),
            events_.TextEnd(block_id=text_id),
        ]
    )

    assert isinstance(out[0], protocol.StartPart)
    assert isinstance(out[1], protocol.StartStepPart)
    assert isinstance(out[2], protocol.TextStartPart) and out[2].id == text_id
    assert isinstance(out[3], protocol.TextDeltaPart) and out[3].delta == "hi"
    assert isinstance(out[4], protocol.TextEndPart) and out[4].id == text_id
    assert isinstance(out[5], protocol.FinishStepPart)
    assert isinstance(out[6], protocol.FinishPart)


async def test_tool_call_and_result_emit_terminal_parts() -> None:
    """ToolCallResult emits tool input and output parts."""
    tool_result_msg = messages_.Message(
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
            # Streaming tool input events from the model
            events_.ToolStart(tool_call_id="tc1", tool_name="search"),
            events_.ToolDelta(tool_call_id="tc1", chunk='{"q":"x"}'),
            events_.ToolEnd(
                tool_call_id="tc1",
                tool_call=messages_.ToolCallPart(
                    tool_call_id="tc1",
                    tool_name="search",
                    tool_args='{"q":"x"}',
                ),
            ),
            # Tool execution result
            agent_events_.ToolCallResult(
                message=tool_result_msg,
                results=tool_result_msg.tool_results,
            ),
        ]
    )
    types = [type(part).__name__ for part in out]
    assert "ToolInputStartPart" in types
    assert "ToolOutputAvailablePart" in types


async def test_tool_result_without_streaming_emits_input_start() -> None:
    """ToolCallResult for a non-streamed tool emits input + output parts."""
    tool_result_msg = messages_.Message(
        role="tool",
        parts=[
            messages_.ToolCallPart(
                id="tc1",
                tool_call_id="tc1",
                tool_name="search",
                tool_args='{"q":"x"}',
            ),
            messages_.ToolResultPart(
                tool_call_id="tc1",
                tool_name="search",
                result={"hits": 1},
            ),
        ],
    )
    out = await _collect(
        [
            agent_events_.ToolCallResult(
                message=tool_result_msg,
                results=tool_result_msg.tool_results,
            ),
        ]
    )
    types = [type(part).__name__ for part in out]
    assert "ToolInputStartPart" in types
    assert "ToolInputAvailablePart" in types
    assert "ToolOutputAvailablePart" in types


async def test_approval_request_hook_emits_approval_part() -> None:
    """HookEvent with pending status emits a ToolApprovalRequestPart."""
    out = await _collect(
        [
            # Streaming tool events first
            events_.ToolStart(tool_call_id="tc1", tool_name="delete"),
            events_.ToolDelta(tool_call_id="tc1", chunk="{}"),
            events_.ToolEnd(
                tool_call_id="tc1",
                tool_call=messages_.ToolCallPart(
                    tool_call_id="tc1",
                    tool_name="delete",
                    tool_args="{}",
                ),
            ),
            # Hook requesting approval
            agent_events_.HookEvent(
                message=messages_.Message(
                    role="internal",
                    parts=[
                        messages_.HookPart(
                            hook_id="approve_tc1",
                            hook_type="ToolApproval",
                            status="pending",
                        )
                    ],
                ),
                hook=messages_.HookPart(
                    hook_id="approve_tc1",
                    hook_type="ToolApproval",
                    status="pending",
                ),
            ),
        ]
    )
    approval_parts = [p for p in out if isinstance(p, protocol.ToolApprovalRequestPart)]
    assert len(approval_parts) == 1
    assert approval_parts[0].tool_call_id == "tc1"
    assert approval_parts[0].approval_id == "approve_tc1"


# NOTE: agent-change boundary detection used to be driven by
# Message.source_label.  That field has been removed; agent-change
# routing in the AI SDK adapter now needs to come from
# PartialToolCallResult, which is a separate piece of work.
