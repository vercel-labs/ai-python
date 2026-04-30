"""Tests for the v3 protocol serialization and deserialization.

Focus areas:
- ``_messages_to_prompt``: the critical outgoing translation layer
- ``_build_request_body``: output_type serialization
- ``_parse_stream_part``: the critical incoming translation layer
- ``_parse_usage``: the two distinct wire formats

Note: tool serialization and provider_options passthrough are tested
end-to-end in ``test_stream.py`` via real HTTP round-trips.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pydantic

from ai.models.ai_gateway import adapter
from ai.types import events as events_
from ai.types import messages

# ---------------------------------------------------------------------------
# _messages_to_prompt
# ---------------------------------------------------------------------------


class TestMessagesToPrompt:
    async def test_system_message(self) -> None:
        msgs = [
            messages.Message(
                role="system",
                parts=[messages.TextPart(text="You are helpful.")],
            )
        ]
        result = await adapter._messages_to_prompt(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    async def test_user_message(self) -> None:
        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Hello")],
            )
        ]
        result = await adapter._messages_to_prompt(msgs)
        assert result == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]

    async def test_assistant_with_reasoning_and_text(self) -> None:
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ReasoningPart(text="Let me think..."),
                    messages.TextPart(text="42"),
                ],
            )
        ]
        result = await adapter._messages_to_prompt(msgs)
        content = result[0]["content"]
        assert content[0] == {"type": "reasoning", "text": "Let me think..."}
        assert content[1] == {"type": "text", "text": "42"}

    async def test_tool_call_with_result_produces_two_messages(self) -> None:
        """A completed tool call must produce an assistant message
        (with the tool-call) AND a tool message (with the result)."""
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolCallPart(
                        tool_call_id="tc-1",
                        tool_name="get_weather",
                        tool_args='{"city": "SF"}',
                    )
                ],
            ),
            messages.Message(
                role="tool",
                parts=[
                    messages.ToolResultPart(
                        tool_call_id="tc-1",
                        tool_name="get_weather",
                        result={"temp": 72},
                    )
                ],
            ),
        ]
        result = await adapter._messages_to_prompt(msgs)
        assert len(result) == 2

        # Assistant message has the tool-call
        tc = result[0]["content"][0]
        assert tc["type"] == "tool-call"
        assert tc["toolCallId"] == "tc-1"
        assert tc["input"] == {"city": "SF"}

        # Tool message has the result
        tr = result[1]["content"][0]
        assert tr["type"] == "tool-result"
        assert tr["output"] == {"type": "json", "value": {"temp": 72}}

    async def test_tool_error_result(self) -> None:
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolCallPart(
                        tool_call_id="tc-1",
                        tool_name="get_weather",
                        tool_args="{}",
                    )
                ],
            ),
            messages.Message(
                role="tool",
                parts=[
                    messages.ToolResultPart(
                        tool_call_id="tc-1",
                        tool_name="get_weather",
                        result="Connection timeout",
                        is_error=True,
                    )
                ],
            ),
        ]
        result = await adapter._messages_to_prompt(msgs)
        tr = result[1]["content"][0]
        assert tr["output"]["type"] == "error-text"
        assert tr["output"]["value"] == "Connection timeout"

    async def test_user_message_with_image_url(self) -> None:
        """FilePart with image URL -> downloaded and converted to data: URL."""
        fake_jpeg = b"\xff\xd8\xff\xe0"
        msgs = [
            messages.Message(
                role="user",
                parts=[
                    messages.TextPart(text="Look at this"),
                    messages.FilePart(
                        data="https://example.com/cat.jpg", media_type="image/jpeg"
                    ),
                ],
            )
        ]
        with patch(
            "ai.models.core.helpers.files.download",
            new_callable=AsyncMock,
            return_value=(fake_jpeg, "image/jpeg"),
        ):
            result = await adapter._messages_to_prompt(msgs)
        content = result[0]["content"]
        assert content[0] == {"type": "text", "text": "Look at this"}
        assert content[1]["type"] == "file"
        assert content[1]["mediaType"] == "image/jpeg"
        assert content[1]["data"].startswith("data:image/jpeg;base64,")

    async def test_user_message_with_file_bytes(self) -> None:
        """FilePart with bytes -> v3 file content part with data URL."""
        msgs = [
            messages.Message(
                role="user",
                parts=[
                    messages.FilePart(
                        data=b"\x89PNG", media_type="image/png", filename="pic.png"
                    ),
                ],
            )
        ]
        result = await adapter._messages_to_prompt(msgs)
        part = result[0]["content"][0]
        assert part["type"] == "file"
        assert part["mediaType"] == "image/png"
        assert part["data"].startswith("data:image/png;base64,")
        assert part["filename"] == "pic.png"

    async def test_pending_tool_call_no_tool_message(self) -> None:
        """A tool call without a corresponding tool-result message
        should NOT produce a tool-result in the prompt."""
        msgs = [
            messages.Message(
                role="assistant",
                parts=[
                    messages.ToolCallPart(
                        tool_call_id="tc-1",
                        tool_name="search",
                        tool_args="{}",
                    )
                ],
            )
        ]
        result = await adapter._messages_to_prompt(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# _build_request_body -- output_type (not tested in test_stream.py)
# ---------------------------------------------------------------------------


class TestBuildRequestBody:
    async def test_with_output_type(self) -> None:
        class WeatherResult(pydantic.BaseModel):
            temp: float
            condition: str

        msgs = [
            messages.Message(
                role="user",
                parts=[messages.TextPart(text="Weather?")],
            )
        ]
        body = await adapter._build_request_body(msgs, output_type=WeatherResult)

        assert "responseFormat" in body
        rf = body["responseFormat"]
        assert rf["type"] == "json"
        assert rf["name"] == "WeatherResult"
        assert "properties" in rf["schema"]
        assert "temp" in rf["schema"]["properties"]


class TestParseStreamPartComplex:
    def test_text_delta_uses_textDelta_key(self) -> None:
        """The gateway sends ``textDelta`` (camelCase), not ``delta``."""
        events = adapter._parse_stream_part(
            {"type": "text-delta", "id": "t1", "textDelta": "Hello"}, set()
        )
        assert isinstance(events[0], events_.TextDelta)
        assert events[0].chunk == "Hello"

    def test_tool_call_expands_to_three_events(self) -> None:
        """A complete ``tool-call`` part must expand into
        ToolStart -> ToolDelta -> ToolEnd."""
        events = adapter._parse_stream_part(
            {
                "type": "tool-call",
                "toolCallId": "tc-1",
                "toolName": "get_weather",
                "input": {"city": "SF"},
            },
            set(),
        )
        assert len(events) == 3
        assert isinstance(events[0], events_.ToolStart)
        assert events[0].tool_name == "get_weather"
        assert isinstance(events[1], events_.ToolDelta)
        assert json.loads(events[1].chunk) == {"city": "SF"}
        assert isinstance(events[2], events_.ToolEnd)

    def test_tool_call_skipped_when_already_streamed(self) -> None:
        """A ``tool-call`` that duplicates a streamed tool is dropped."""
        seen: set[str] = set()
        adapter._parse_stream_part(
            {"type": "tool-input-start", "id": "tc-1", "toolName": "get_weather"},
            seen,
        )
        events = adapter._parse_stream_part(
            {
                "type": "tool-call",
                "toolCallId": "tc-1",
                "toolName": "get_weather",
                "input": {"city": "SF"},
            },
            seen,
        )
        assert events == []

    def test_finish_flat_usage(self) -> None:
        events = adapter._parse_stream_part(
            {
                "type": "finish",
                "finishReason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                },
            },
            set(),
        )
        done = events[0]
        assert isinstance(done, events_.StreamEnd)
        assert done.usage is not None
        assert done.usage.input_tokens == 10
        assert done.usage.output_tokens == 20

    def test_finish_v3_nested_usage(self) -> None:
        events = adapter._parse_stream_part(
            {
                "type": "finish",
                "finishReason": {
                    "unified": "tool-calls",
                    "raw": "tool_calls",
                },
                "usage": {
                    "inputTokens": {
                        "total": 100,
                        "cacheRead": 50,
                    },
                    "outputTokens": {
                        "total": 200,
                        "reasoning": 30,
                    },
                },
            },
            set(),
        )
        done = events[0]
        assert isinstance(done, events_.StreamEnd)
        assert done.usage is not None
        assert done.usage.input_tokens == 100
        assert done.usage.cache_read_tokens == 50
        assert done.usage.reasoning_tokens == 30

    def test_file_part(self) -> None:
        """A ``file`` stream part (inline image from Gemini/GPT-5)
        must produce a FileEvent."""
        events = adapter._parse_stream_part(
            {
                "type": "file",
                "id": "f1",
                "mediaType": "image/png",
                "data": "iVBORw0KGgo=",
            },
            set(),
        )
        assert len(events) == 1
        assert isinstance(events[0], events_.FileEvent)
        assert events[0].block_id == "f1"
        assert events[0].media_type == "image/png"
        assert events[0].data == "iVBORw0KGgo="

    def test_file_part_defaults(self) -> None:
        """A minimal ``file`` part uses sensible defaults."""
        events = adapter._parse_stream_part({"type": "file", "data": "somedata"}, set())
        assert len(events) == 1
        assert isinstance(events[0], events_.FileEvent)
        assert events[0].media_type == "application/octet-stream"

    def test_unknown_types_produce_no_events(self) -> None:
        for t in ("stream-start", "raw", "response-metadata", "banana"):
            assert adapter._parse_stream_part({"type": t}, set()) == []


# ---------------------------------------------------------------------------
# _parse_usage
# ---------------------------------------------------------------------------


class TestParseUsage:
    def test_flat_format(self) -> None:
        usage = adapter._parse_usage({"prompt_tokens": 10, "completion_tokens": 20})
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20

    def test_v3_nested_format(self) -> None:
        usage = adapter._parse_usage(
            {
                "inputTokens": {
                    "total": 100,
                    "cacheRead": 30,
                    "cacheWrite": 5,
                },
                "outputTokens": {"total": 50, "reasoning": 10},
            }
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 30
        assert usage.cache_write_tokens == 5
        assert usage.reasoning_tokens == 10

    def test_non_dict_returns_empty(self) -> None:
        usage = adapter._parse_usage("not a dict")
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
