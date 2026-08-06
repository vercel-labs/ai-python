"""Tests for the v3 protocol serialization and deserialization.

Focus areas:
- ``v3._messages_to_prompt``: the critical outgoing translation layer
- ``v3._parse_stream_part``: the critical incoming translation layer
- ``v3._tool_result_output`` and the inline file encoding
- the thinking-signature round trip

Version-independent pieces (params mapping, usage parsing, media
endpoints) are tested in ``test_shared.py``; end-to-end request/response
behavior in ``../test_stream.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

from ai import models
from ai.providers.ai_gateway.protocol import v3
from ai.types import events as events_
from ai.types import messages

# ---------------------------------------------------------------------------
# _messages_to_prompt
# ---------------------------------------------------------------------------


async def test_system_message() -> None:
    msgs = [
        messages.Message(
            role="system",
            parts=[messages.TextPart(text="You are helpful.")],
        )
    ]
    result = await v3._messages_to_prompt(msgs)
    assert result == [{"role": "system", "content": "You are helpful."}]


async def test_user_message() -> None:
    msgs = [
        messages.Message(
            role="user",
            parts=[messages.TextPart(text="Hello")],
        )
    ]
    result = await v3._messages_to_prompt(msgs)
    assert result == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }
    ]


async def test_assistant_with_reasoning_and_text() -> None:
    msgs = [
        messages.Message(
            role="assistant",
            parts=[
                messages.ReasoningPart(text="Let me think..."),
                messages.TextPart(text="42"),
            ],
        )
    ]
    result = await v3._messages_to_prompt(msgs)
    content = result[0]["content"]
    assert content[0] == {"type": "reasoning", "text": "Let me think..."}
    assert content[1] == {"type": "text", "text": "42"}


async def test_assistant_reasoning_replays_signature() -> None:
    """A reasoning part's metadata (the thinking-block signature) must
    be replayed verbatim as ``providerOptions`` so the upstream can
    verify its own thinking."""
    msgs = [
        messages.Message(
            role="assistant",
            parts=[
                messages.ReasoningPart(
                    text="Let me think...",
                    provider_metadata={
                        "anthropic": {"signature": "ErMJabc123"}
                    },
                ),
            ],
        )
    ]
    result = await v3._messages_to_prompt(msgs)
    assert result[0]["content"][0] == {
        "type": "reasoning",
        "text": "Let me think...",
        "providerOptions": {"anthropic": {"signature": "ErMJabc123"}},
    }


async def test_assistant_reasoning_without_signature_omits_options() -> None:
    """No signature -> no ``providerOptions`` key (back-compat)."""
    msgs = [
        messages.Message(
            role="assistant",
            parts=[messages.ReasoningPart(text="hmm")],
        )
    ]
    result = await v3._messages_to_prompt(msgs)
    assert result[0]["content"][0] == {"type": "reasoning", "text": "hmm"}


async def test_tool_call_with_result_produces_two_messages() -> None:
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
    result = await v3._messages_to_prompt(msgs)
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


async def test_tool_error_result() -> None:
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
                    result_kind="error",
                )
            ],
        ),
    ]
    result = await v3._messages_to_prompt(msgs)
    tr = result[1]["content"][0]
    assert tr["output"]["type"] == "error-text"
    assert tr["output"]["value"] == "Connection timeout"


async def test_user_message_with_image_url() -> None:
    """FilePart with image URL -> downloaded and converted to data: URL."""
    fake_jpeg = b"\xff\xd8\xff\xe0"
    msgs = [
        messages.Message(
            role="user",
            parts=[
                messages.TextPart(text="Look at this"),
                messages.FilePart(
                    data="https://example.com/cat.jpg",
                    media_type="image/jpeg",
                ),
            ],
        )
    ]
    with patch(
        "ai.models.core.helpers.files.download",
        new_callable=AsyncMock,
        return_value=(fake_jpeg, "image/jpeg"),
    ):
        result = await v3._messages_to_prompt(msgs)
    content = result[0]["content"]
    assert content[0] == {"type": "text", "text": "Look at this"}
    assert content[1]["type"] == "file"
    assert content[1]["mediaType"] == "image/jpeg"
    assert content[1]["data"].startswith("data:image/jpeg;base64,")


async def test_user_message_with_file_bytes() -> None:
    """FilePart with bytes -> v3 file content part with data URL."""
    msgs = [
        messages.Message(
            role="user",
            parts=[
                messages.FilePart(
                    data=b"\x89PNG",
                    media_type="image/png",
                    filename="pic.png",
                ),
            ],
        )
    ]
    result = await v3._messages_to_prompt(msgs)
    part = result[0]["content"][0]
    assert part["type"] == "file"
    assert part["mediaType"] == "image/png"
    assert part["data"].startswith("data:image/png;base64,")
    assert part["filename"] == "pic.png"


async def test_repairs_history() -> None:
    """Conversion runs history_utils.repair: internal messages are
    dropped and orphaned tool calls get a synthetic error result."""
    msgs = [
        messages.Message(
            role="internal",
            parts=[messages.TextPart(text="app-only")],
        ),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="tc-1",
                    tool_name="search",
                    tool_args="{}",
                )
            ],
        ),
    ]
    result = await v3._messages_to_prompt(msgs)
    assert len(result) == 2
    assert result[0]["role"] == "assistant"
    assert result[1]["role"] == "tool"
    (tool_result,) = result[1]["content"]
    assert tool_result["toolCallId"] == "tc-1"
    assert tool_result["output"]["type"] == "error-text"


# ---------------------------------------------------------------------------
# _parse_stream_part
# ---------------------------------------------------------------------------


def test_text_delta_uses_text_delta_key() -> None:
    """The gateway sends ``textDelta`` (camelCase), not ``delta``."""
    events = v3._parse_stream_part(
        {"type": "text-delta", "id": "t1", "textDelta": "Hello"}, set()
    )
    assert isinstance(events[0], events_.TextDelta)
    assert events[0].chunk == "Hello"


def test_tool_call_expands_to_three_events() -> None:
    """A complete ``tool-call`` part must expand into
    ToolStart -> ToolDelta -> ToolEnd."""
    events = v3._parse_stream_part(
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


def test_tool_call_skipped_when_already_streamed() -> None:
    """A ``tool-call`` that duplicates a streamed tool is dropped."""
    seen: set[str] = set()
    v3._parse_stream_part(
        {
            "type": "tool-input-start",
            "id": "tc-1",
            "toolName": "get_weather",
        },
        seen,
    )
    events = v3._parse_stream_part(
        {
            "type": "tool-call",
            "toolCallId": "tc-1",
            "toolName": "get_weather",
            "input": {"city": "SF"},
        },
        seen,
    )
    assert events == []


def test_finish_flat_usage() -> None:
    events = v3._parse_stream_part(
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


def test_finish_v3_nested_usage() -> None:
    events = v3._parse_stream_part(
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


def test_reasoning_delta_carries_provider_metadata() -> None:
    """A reasoning-delta's ``providerMetadata`` (the thinking-block
    signature) rides through verbatim on ``provider_metadata``."""
    events = v3._parse_stream_part(
        {
            "type": "reasoning-delta",
            "id": "0",
            "delta": "",
            "providerMetadata": {"anthropic": {"signature": "ErMJabc123"}},
        },
        set(),
    )
    assert len(events) == 1
    delta = events[0]
    assert isinstance(delta, events_.ReasoningDelta)
    assert delta.provider_metadata == {"anthropic": {"signature": "ErMJabc123"}}


def test_reasoning_delta_without_metadata() -> None:
    """A plain reasoning-delta carries no provider_metadata."""
    events = v3._parse_stream_part(
        {"type": "reasoning-delta", "id": "0", "delta": "thinking"},
        set(),
    )
    assert isinstance(events[0], events_.ReasoningDelta)
    assert events[0].provider_metadata is None


def test_reasoning_start_drops_routing_metadata() -> None:
    """Metadata on -start is gateway routing info (generationId), not
    provider reasoning metadata, and must not be replayed."""
    events = v3._parse_stream_part(
        {
            "type": "reasoning-start",
            "id": "0",
            "providerMetadata": {"gateway": {"generationId": "gen_1"}},
        },
        set(),
    )
    assert isinstance(events[0], events_.ReasoningStart)
    assert events[0].provider_metadata is None


def test_file_part() -> None:
    """A ``file`` stream part (inline image from Gemini/GPT-5)
    must produce a FileEvent."""
    events = v3._parse_stream_part(
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


def test_file_part_defaults() -> None:
    """A minimal ``file`` part uses sensible defaults."""
    events = v3._parse_stream_part({"type": "file", "data": "somedata"}, set())
    assert len(events) == 1
    assert isinstance(events[0], events_.FileEvent)
    assert events[0].media_type == "application/octet-stream"


def test_unknown_types_produce_no_events() -> None:
    for t in ("stream-start", "raw", "response-metadata", "banana"):
        assert v3._parse_stream_part({"type": t}, set()) == []


# ---------------------------------------------------------------------------
# Multi-part tool result helpers
# ---------------------------------------------------------------------------


def test_inline_image_data() -> None:
    fp = messages.FilePart(data="b64data", media_type="image/png")
    entry = v3._file_part_to_inline(fp)
    assert entry == {
        "type": "image-data",
        "data": "b64data",
        "mediaType": "image/png",
    }


def test_inline_file_data_with_filename() -> None:
    fp = messages.FilePart(
        data="pdfdata",
        media_type="application/pdf",
        filename="doc.pdf",
    )
    entry = v3._file_part_to_inline(fp)
    assert entry["type"] == "file-data"
    assert entry["mediaType"] == "application/pdf"
    assert entry["filename"] == "doc.pdf"


def test_inline_bytes_become_base64() -> None:
    fp = messages.FilePart(data=b"\x89PNG", media_type="image/png")
    entry = v3._file_part_to_inline(fp)
    assert entry["type"] == "image-data"
    assert entry["data"] != ""


def _tool_result_part(
    result: object, *, result_kind: messages.ResultKind = "json"
) -> messages.ToolResultPart:
    return messages.ToolResultPart(
        tool_call_id="tc-1",
        tool_name="t",
        result=result,
        result_kind=result_kind,
    )


def test_tool_result_output_text() -> None:
    result = v3._tool_result_output(_tool_result_part("hi"))
    assert result == {"type": "text", "value": "hi"}


def test_tool_result_output_json() -> None:
    result = v3._tool_result_output(_tool_result_part({"key": "value"}))
    assert result == {"type": "json", "value": {"key": "value"}}


def test_tool_result_output_error_text() -> None:
    result = v3._tool_result_output(
        _tool_result_part("oops", result_kind="error")
    )
    assert result == {"type": "error-text", "value": "oops"}


def test_tool_result_output_error_json() -> None:
    result = v3._tool_result_output(
        _tool_result_part({"code": 500}, result_kind="error")
    )
    assert result == {"type": "error-json", "value": {"code": 500}}


def test_tool_result_output_content_multipart() -> None:
    fp = messages.FilePart(data="b64", media_type="image/jpeg")
    result = v3._tool_result_output(
        _tool_result_part(
            messages.ContentOutput(value=[messages.TextPart(text="desc"), fp]),
            result_kind="special",
        )
    )
    assert result["type"] == "content"
    assert result["value"][0] == {"type": "text", "text": "desc"}
    assert result["value"][1]["type"] == "image-data"


async def test_tool_result_with_file_part() -> None:
    """ContentOutput with a FilePart uses the 'content' wire output."""
    fp = messages.FilePart(data="iVBOR", media_type="image/png")
    msgs = [
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="tc-1",
                    tool_name="read",
                    tool_args='{"path": "test.png"}',
                )
            ],
        ),
        messages.Message(
            role="tool",
            parts=[
                messages.ToolResultPart(
                    tool_call_id="tc-1",
                    tool_name="read",
                    result=messages.ContentOutput(
                        value=[
                            messages.TextPart(text="Image loaded"),
                            fp,
                        ]
                    ),
                )
            ],
        ),
    ]
    result = await v3._messages_to_prompt(msgs)
    tr = result[1]["content"][0]
    assert tr["output"]["type"] == "content"
    assert tr["output"]["value"][0] == {
        "type": "text",
        "text": "Image loaded",
    }
    assert tr["output"]["value"][1] == {
        "type": "image-data",
        "data": "iVBOR",
        "mediaType": "image/png",
    }


# ---------------------------------------------------------------------------
# Thinking-block round trip (signature survives in -> aggregate -> out)
# ---------------------------------------------------------------------------


async def test_signature_survives_round_trip() -> None:
    """The whole point of capturing the signature: it must survive being
    parsed from the wire, aggregated into a Message, and re-serialized so
    the upstream sees its own thinking on the next turn."""
    # Wire parts as the gateway emits them: the signature rides on the
    # final (empty) reasoning-delta, not the start or end.
    wire_parts: list[dict[str, Any]] = [
        {"type": "reasoning-start", "id": "0"},
        {"type": "reasoning-delta", "id": "0", "delta": "thinking hard"},
        {
            "type": "reasoning-delta",
            "id": "0",
            "delta": "",
            "providerMetadata": {"anthropic": {"signature": "ErMJsig=="}},
        },
        {"type": "reasoning-end", "id": "0"},
        {"type": "finish"},
    ]

    async def _gen() -> AsyncGenerator[events_.Event]:
        for part in wire_parts:
            for event in v3._parse_stream_part(part, set()):
                yield event

    stream = models.Stream(_gen())
    async for _ in stream:
        pass

    # Aggregated message: one reasoning part carrying the signature.
    reasoning = [
        p for p in stream.message.parts if isinstance(p, messages.ReasoningPart)
    ]
    assert len(reasoning) == 1
    assert reasoning[0].text == "thinking hard"
    assert reasoning[0].provider_metadata == {
        "anthropic": {"signature": "ErMJsig=="}
    }

    # Round-trip back out: the metadata is replayed verbatim to the
    # provider as providerOptions.
    out = await v3._messages_to_prompt(
        [messages.Message(role="assistant", parts=stream.message.parts)]
    )
    assert out[0]["content"][0]["providerOptions"] == {
        "anthropic": {"signature": "ErMJsig=="}
    }
