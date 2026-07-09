from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import ai
from ai.agents.ui.ai_sdk import to_sse, to_stream, ui_events
from ai.agents.ui.ai_sdk.outbound_stream import (
    format_done_sse,
    format_sse,
    serialize_event,
)
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
) -> list[ui_events.UIMessageStreamEvent]:
    return [event async for event in to_stream(_gen(stream_events))]


def _source_messages(metadata: Any | None) -> list[dict[str, Any]]:
    assert isinstance(metadata, dict)
    adapter_metadata = metadata.get("aiPython")
    assert isinstance(adapter_metadata, dict)
    source_messages = adapter_metadata.get("sourceMessages")
    assert isinstance(source_messages, list)
    return source_messages


def test_serialize_event_camelcases_keys() -> None:
    event = ui_events.UIStartEvent(message_id="m1")
    payload = json.loads(serialize_event(event))
    assert payload == {"type": "start", "messageId": "m1"}


def test_format_sse_wraps_data_line() -> None:
    event = ui_events.UITextDeltaEvent(id="t1", delta="hi")
    line = format_sse(event)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")


def test_serialize_data_event_uses_type_with_prefix() -> None:
    event = ui_events.UIDataEvent(data_type="custom", data={"k": 1})
    payload = json.loads(serialize_event(event))
    assert payload["type"] == "data-custom"
    assert "dataType" not in payload


def test_serialize_protocol_fields_use_ai_sdk_wire_names() -> None:
    event = ui_events.UIToolApprovalResponseEvent(
        approval_id="approval-1",
        approved=False,
        reason="no",
        provider_executed=True,
        provider_metadata={"provider": {"k": "v"}},
    )

    payload = json.loads(serialize_event(event))

    assert payload == {
        "type": "tool-approval-response",
        "approvalId": "approval-1",
        "approved": False,
        "reason": "no",
        "providerExecuted": True,
        "providerMetadata": {"provider": {"k": "v"}},
    }


def test_format_done_sse_returns_done_sentinel() -> None:
    assert format_done_sse() == "data: [DONE]\n\n"


async def test_to_sse_emits_data_prefixed_lines() -> None:
    lines = [
        line
        async for line in to_sse(
            _gen(
                [
                    events_.TextStart(block_id="t1"),
                    events_.TextDelta(block_id="t1", chunk="hi"),
                    events_.TextEnd(block_id="t1"),
                ]
            )
        )
    ]
    assert all(line.startswith("data: ") for line in lines)
    first = json.loads(lines[0].removeprefix("data: ").rstrip())
    assert first["type"] == "start"
    assert lines[-1] == "data: [DONE]\n\n"


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

    start = next(
        event for event in out if isinstance(event, ui_events.UIStartEvent)
    )
    assert start.message_id == "assistant-runtime-id"


async def test_finish_metadata_tracks_streamed_assistant_message() -> None:
    assistant = messages_.Message(
        id="assistant-1",
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

    finish = next(
        event for event in out if isinstance(event, ui_events.UIFinishEvent)
    )
    assert _source_messages(finish.message_metadata) == [
        {
            "id": "assistant-1",
            "role": "assistant",
            "turnId": None,
            "partIds": ["text-1"],
        }
    ]


async def test_finish_metadata_tracks_tool_and_internal_messages() -> None:
    tool_call = messages_.ToolCallPart(
        id="call-1",
        tool_call_id="tc1",
        tool_name="search",
        tool_args="{}",
    )
    assistant = messages_.Message(
        id="assistant-1",
        turn_id="turn-1",
        role="assistant",
        parts=[tool_call],
    )
    tool = messages_.Message(
        id="tool-1",
        turn_id="turn-1",
        role="tool",
        parts=[
            messages_.ToolResultPart(
                id="result-1",
                tool_call_id="tc1",
                tool_name="search",
                result={"hits": 1},
            )
        ],
    )
    hook = messages_.HookPart[Any](
        id="hook-1",
        hook_id="approve_tc1",
        hook_type="ToolApproval",
        status="deferred",
    )
    internal = messages_.Message(
        id="internal-1",
        turn_id="turn-1",
        role="internal",
        parts=[hook],
    )

    out = await _collect(
        [
            events_.ToolStart(
                tool_call_id="tc1", tool_name="search", message=assistant
            ),
            events_.ToolEnd(
                tool_call_id="tc1", tool_call=tool_call, message=assistant
            ),
            agent_events_.ToolCallResult(
                message=tool,
                results=tool.tool_results,
            ),
            agent_events_.HookEvent(message=internal, hook=hook),
        ]
    )

    finish = next(
        event for event in out if isinstance(event, ui_events.UIFinishEvent)
    )
    assert [
        source["id"] for source in _source_messages(finish.message_metadata)
    ] == ["assistant-1", "tool-1", "internal-1"]


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

    assert isinstance(out[0], ui_events.UIStartEvent)
    assert isinstance(out[1], ui_events.UIStartStepEvent)
    assert (
        isinstance(out[2], ui_events.UITextStartEvent) and out[2].id == text_id
    )
    assert (
        isinstance(out[3], ui_events.UITextDeltaEvent) and out[3].delta == "hi"
    )
    assert isinstance(out[4], ui_events.UITextEndEvent) and out[4].id == text_id
    assert isinstance(out[5], ui_events.UIFinishStepEvent)
    assert isinstance(out[6], ui_events.UIFinishEvent)
    assert out[6].message_metadata is None


async def test_finish_metadata_ignores_empty_messages() -> None:
    assistant = messages_.Message(
        id="assistant-empty",
        role="assistant",
        parts=[],
    )

    out = await _collect(
        [
            events_.StreamStart(message=assistant),
            events_.StreamEnd(message=assistant),
        ]
    )

    finish = next(
        event for event in out if isinstance(event, ui_events.UIFinishEvent)
    )
    assert finish.message_metadata is None


async def test_tool_call_and_result_emit_terminal_events() -> None:
    """ToolCallResult emits tool input and output events."""
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
    types = [type(event).__name__ for event in out]
    assert "UIToolInputStartEvent" in types
    assert "UIToolOutputAvailableEvent" in types


async def test_tool_result_without_streaming_emits_input_start() -> None:
    """ToolCallResult for a non-streamed tool emits input + output events."""
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
    types = [type(event).__name__ for event in out]
    assert "UIToolInputStartEvent" in types
    assert "UIToolInputAvailableEvent" in types
    assert "UIToolOutputAvailableEvent" in types


async def test_approval_request_hook_emits_approval_event() -> None:
    """HookEvent with deferred status emits a UIToolApprovalRequestEvent."""
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
                            status="deferred",
                            tool_call_id="tc1",
                        )
                    ],
                ),
                hook=messages_.HookPart(
                    hook_id="approve_tc1",
                    hook_type="ToolApproval",
                    status="deferred",
                    tool_call_id="tc1",
                ),
            ),
        ]
    )
    approval_events = [
        p for p in out if isinstance(p, ui_events.UIToolApprovalRequestEvent)
    ]
    assert len(approval_events) == 1
    assert approval_events[0].tool_call_id == "tc1"
    assert approval_events[0].approval_id == "approve_tc1"


async def test_partial_tool_results_emit_preliminary_outputs() -> None:
    """Each partial result yields a preliminary event."""
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
        if isinstance(p, ui_events.UIToolOutputAvailableEvent) and p.preliminary
    ]
    assert [p.output for p in prelim] == [
        "hit 1, ",
        "hit 1, hit 2, ",
        "hit 1, hit 2, hit 3",
    ]
    assert all(p.tool_call_id == "tc1" for p in prelim)


async def test_partial_message_bundle_becomes_single_ui_message() -> None:
    """A one-bubble MessageAggregator snapshot serializes to one UIMessage.

    Matches the AI SDK sub-agent convention: a single ``UIMessage`` (not a
    one-element list) for the common case.
    """
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
        if isinstance(p, ui_events.UIToolOutputAvailableEvent) and p.preliminary
    ]
    assert isinstance(prelim.output, dict)
    assert prelim.output["role"] == "assistant"
    assert prelim.output["parts"][0]["type"] == "text"


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
    assert not any(
        isinstance(p, ui_events.UIToolOutputAvailableEvent) for p in out
    )


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

    start = next(
        p for p in out if isinstance(p, ui_events.UIToolInputStartEvent)
    )
    assert start.provider_executed is True
    assert start.dynamic is True
    assert start.provider_metadata == {"provider": {"start": True}}

    available = next(
        p for p in out if isinstance(p, ui_events.UIToolInputAvailableEvent)
    )
    assert available.provider_executed is True
    assert available.dynamic is True
    assert available.input == {"q": "ai"}
    assert available.provider_metadata == {"provider": {"call": True}}

    result = next(
        p for p in out if isinstance(p, ui_events.UIToolOutputAvailableEvent)
    )
    assert result.provider_executed is True
    assert result.dynamic is True
    assert result.output == {"hits": 1}
    assert result.provider_metadata == {"provider": {"result": True}}


async def test_file_event_emits_ui_file_event() -> None:
    out = await _collect(
        [
            events_.FileEvent(
                media_type="image/png",
                data=b"abc",
                provider_metadata={"provider": {"file": True}},
            )
        ]
    )

    file_event = next(p for p in out if isinstance(p, ui_events.UIFileEvent))
    assert file_event.url == "data:image/png;base64,YWJj"
    assert file_event.media_type == "image/png"
    assert file_event.provider_metadata == {"provider": {"file": True}}


async def test_resolved_approval_hook_emits_response_event() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="approve_tc1",
        hook_type="ToolApproval",
        status="resolved",
        metadata={
            "providerExecuted": True,
            "callProviderMetadata": {"provider": {"approval": True}},
        },
        resolution={"granted": False, "reason": "not allowed"},
        tool_call_id="tc1",
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
        p for p in out if isinstance(p, ui_events.UIToolApprovalResponseEvent)
    )
    assert response.approval_id == "approve_tc1"
    assert response.approved is False
    assert response.reason == "not allowed"
    assert response.provider_executed is True
    assert response.provider_metadata == {"provider": {"approval": True}}
    assert any(isinstance(p, ui_events.UIToolOutputDeniedEvent) for p in out)


# NOTE: agent-change boundary detection used to be driven by
# Message.source_label.  That field has been removed; agent-change
# routing in the AI SDK adapter now needs to come from
# PartialToolCallResult, which is a separate piece of work.
