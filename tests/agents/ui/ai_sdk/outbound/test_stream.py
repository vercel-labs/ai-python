from __future__ import annotations

from collections.abc import AsyncGenerator

import ai
from ai.agents.ui.ai_sdk import protocol, to_stream
from ai.types import events as agent_events_
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


async def test_stream_start_uses_runtime_message_id() -> None:
    assistant = messages_.Message(
        id="assistant-runtime-id",
        role="assistant",
        parts=[messages_.TextPart(id="text-1", text="hello")],
    )

    out = await _collect(
        [
            events_.TextStart(block_id="text-1", message=assistant),
            events_.TextDelta(
                block_id="text-1", chunk="hello", message=assistant
            ),
            events_.TextEnd(block_id="text-1", message=assistant),
        ]
    )

    start = next(part for part in out if isinstance(part, protocol.StartPart))
    assert start.message_id == "assistant-runtime-id"


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
    approval_parts = [
        p for p in out if isinstance(p, protocol.ToolApprovalRequestPart)
    ]
    assert len(approval_parts) == 1
    assert approval_parts[0].tool_call_id == "tc1"
    assert approval_parts[0].approval_id == "approve_tc1"


async def test_partial_tool_results_emit_preliminary_outputs() -> None:
    """Each partial result yields a preliminary part."""
    out = await _collect(
        [
            agent_events_.PartialToolCallResult(
                tool_call_id="tc1",
                tool_name="search",
                value="hit 1, ",
                aggregator_factory=ai.agents.ConcatAggregator,
            ),
            agent_events_.PartialToolCallResult(
                tool_call_id="tc1",
                tool_name="search",
                value="hit 2, ",
                aggregator_factory=ai.agents.ConcatAggregator,
            ),
            agent_events_.PartialToolCallResult(
                tool_call_id="tc1",
                tool_name="search",
                value="hit 3",
                aggregator_factory=ai.agents.ConcatAggregator,
            ),
        ]
    )

    prelim = [
        p
        for p in out
        if isinstance(p, protocol.ToolOutputAvailablePart) and p.preliminary
    ]
    assert [p.output for p in prelim] == [
        "hit 1, ",
        "hit 1, hit 2, ",
        "hit 1, hit 2, hit 3",
    ]
    assert all(p.tool_call_id == "tc1" for p in prelim)


async def test_partial_message_bundle_becomes_ui_message() -> None:
    """MessageAggregator's snapshot collapses to one UIMessage."""
    from ai.agents.ui.ai_sdk.ui_message import UIMessage

    inner_msg = messages_.Message(
        role="assistant",
        parts=[messages_.TextPart(text="hi from sub-agent")],
    )

    out = await _collect(
        [
            agent_events_.PartialToolCallResult(
                tool_call_id="tc1",
                tool_name="research",
                value=agent_events_.ToolCallResult(
                    message=inner_msg, results=[]
                ),
                aggregator_factory=ai.agents.MessageAggregator,
            ),
        ]
    )

    [prelim] = [
        p
        for p in out
        if isinstance(p, protocol.ToolOutputAvailablePart) and p.preliminary
    ]
    assert isinstance(prelim.output, UIMessage)
    assert prelim.output.role == "assistant"
    assert prelim.output.parts[0].type == "text"


async def test_partial_tool_result_without_factory_is_skipped() -> None:
    """Without an aggregator_factory there's nothing to snapshot."""
    out = await _collect(
        [
            agent_events_.PartialToolCallResult(
                tool_call_id="tc1",
                tool_name="search",
                value="ignored",
            ),
        ]
    )
    assert not any(isinstance(p, protocol.ToolOutputAvailablePart) for p in out)


async def test_builtin_tool_stream_marks_provider_executed_dynamic() -> None:
    out = await _collect(
        [
            events_.BuiltinToolStart(
                tool_call_id="tc1",
                tool_name="web_search",
                provider_metadata={"provider": {"start": True}},
            ),
            events_.BuiltinToolDelta(tool_call_id="tc1", chunk='{"q":"ai"}'),
            events_.BuiltinToolEnd(
                tool_call_id="tc1",
                tool_call=messages_.BuiltinToolCallPart(
                    tool_call_id="tc1",
                    tool_name="web_search",
                    tool_args='{"q":"ai"}',
                    provider_metadata={"provider": {"call": True}},
                ),
            ),
            events_.BuiltinToolResult(
                tool_call_id="tc1",
                result=messages_.BuiltinToolReturnPart(
                    tool_call_id="tc1",
                    tool_name="web_search",
                    result={"hits": 1},
                    provider_metadata={"provider": {"result": True}},
                ),
            ),
        ]
    )

    start = next(p for p in out if isinstance(p, protocol.ToolInputStartPart))
    assert start.provider_executed is True
    assert start.dynamic is True
    assert start.provider_metadata == {"provider": {"start": True}}

    available = next(
        p for p in out if isinstance(p, protocol.ToolInputAvailablePart)
    )
    assert available.provider_executed is True
    assert available.dynamic is True
    assert available.input == {"q": "ai"}
    assert available.provider_metadata == {"provider": {"call": True}}

    result = next(
        p for p in out if isinstance(p, protocol.ToolOutputAvailablePart)
    )
    assert result.provider_executed is True
    assert result.dynamic is True
    assert result.output == {"hits": 1}
    assert result.provider_metadata == {"provider": {"result": True}}


async def test_file_event_emits_file_part_with_data_url_and_metadata() -> None:
    out = await _collect(
        [
            events_.FileEvent(
                media_type="image/png",
                data=b"abc",
                provider_metadata={"provider": {"file": True}},
            )
        ]
    )

    file_part = next(p for p in out if isinstance(p, protocol.FilePart))
    assert file_part.url == "data:image/png;base64,YWJj"
    assert file_part.media_type == "image/png"
    assert file_part.provider_metadata == {"provider": {"file": True}}


async def test_resolved_approval_hook_emits_response_part() -> None:
    hook = messages_.HookPart(
        hook_id="approve_tc1",
        hook_type="ToolApproval",
        status="resolved",
        metadata={
            "providerExecuted": True,
            "callProviderMetadata": {"provider": {"approval": True}},
        },
        resolution={"granted": False, "reason": "not allowed"},
    )

    out = await _collect(
        [
            agent_events_.HookEvent(
                message=messages_.Message(
                    id="turn-1:internal:0",
                    turn_id="turn-1",
                    role="internal",
                    parts=[hook],
                ),
                hook=hook,
            )
        ]
    )

    response = next(
        p for p in out if isinstance(p, protocol.ToolApprovalResponsePart)
    )
    assert response.approval_id == "approve_tc1"
    assert response.approved is False
    assert response.reason == "not allowed"
    assert response.provider_executed is True
    assert response.provider_metadata == {"provider": {"approval": True}}
    assert any(isinstance(p, protocol.ToolOutputDeniedPart) for p in out)


# NOTE: agent-change boundary detection used to be driven by
# Message.source_label.  That field has been removed; agent-change
# routing in the AI SDK adapter now needs to come from
# PartialToolCallResult, which is a separate piece of work.
