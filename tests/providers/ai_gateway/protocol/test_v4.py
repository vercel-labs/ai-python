"""Tests for the v4 protocol serialization and deserialization.

Focus areas:
- ``v4._messages_to_prompt``: tagged file-data unions, URL passthrough
- ``v4._apply_reasoning``: the standardized effort field
- ``v4._parse_stream_part``: object finish reasons, tagged file data,
  response metadata, error parts, warnings

Version-independent pieces are tested in ``test_shared.py``; end-to-end
request/response behavior in ``../test_stream.py``.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from ai import models
from ai.providers.ai_gateway.client import errors as client_errors
from ai.providers.ai_gateway.protocol import v4
from ai.types import events as events_
from ai.types import messages

# ---------------------------------------------------------------------------
# _messages_to_prompt, v4 encoding
# ---------------------------------------------------------------------------


async def test_v4_user_message_with_image_url() -> None:
    """v4: FilePart with an http URL passes through, no download."""
    msgs = [
        messages.Message(
            role="user",
            parts=[
                messages.FilePart(
                    data="https://example.com/cat.jpg",
                    media_type="image/jpeg",
                ),
            ],
        )
    ]
    result = await v4._messages_to_prompt(msgs)
    part = result[0]["content"][0]
    assert part == {
        "type": "file",
        "mediaType": "image/jpeg",
        "data": {"type": "url", "url": "https://example.com/cat.jpg"},
    }


async def test_v4_user_message_with_file_bytes() -> None:
    """v4: FilePart with bytes -> tagged inline data, not a data URL."""
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
    result = await v4._messages_to_prompt(msgs)
    part = result[0]["content"][0]
    assert part == {
        "type": "file",
        "mediaType": "image/png",
        "data": {
            "type": "data",
            "data": base64.b64encode(b"\x89PNG").decode(),
        },
        "filename": "pic.png",
    }


async def test_v4_user_message_with_data_url() -> None:
    """v4: a data: URL is unwrapped to tagged inline data."""
    b64 = base64.b64encode(b"\x89PNG").decode()
    msgs = [
        messages.Message(
            role="user",
            parts=[
                messages.FilePart(
                    data=f"data:image/png;base64,{b64}",
                    media_type="image/png",
                ),
            ],
        )
    ]
    result = await v4._messages_to_prompt(msgs)
    part = result[0]["content"][0]
    assert part["data"] == {"type": "data", "data": b64}


# ---------------------------------------------------------------------------
# _apply_reasoning
# ---------------------------------------------------------------------------


def test_reasoning_effort_is_plain_string() -> None:
    """v4 standardizes ``reasoning`` as an effort level, not an object."""
    body: dict[str, Any] = {}
    v4._apply_reasoning(
        body,
        models.InferenceRequestParams(
            reasoning=models.ReasoningParams(effort="high")
        ),
        provider="test-provider",
    )
    assert body["reasoning"] == "high"


def test_reasoning_effort_none_maps_to_none_level() -> None:
    body: dict[str, Any] = {}
    v4._apply_reasoning(
        body,
        models.InferenceRequestParams(
            reasoning=models.ReasoningParams(effort=None)
        ),
        provider="test-provider",
    )
    assert body["reasoning"] == "none"


def test_reasoning_summary_requires_dedicated_provider() -> None:
    with pytest.raises(ValueError, match="reasoning summary"):
        v4._apply_reasoning(
            {},
            models.InferenceRequestParams(
                output=models.OutputParams(reasoning_summary="detailed")
            ),
            provider="test-provider",
        )


def test_finish_object_reason() -> None:
    """v4 sends ``finishReason`` as {unified, raw}; raw is preserved."""
    events = v4._parse_stream_part(
        {
            "type": "finish",
            "finishReason": {"unified": "stop", "raw": "end_turn"},
        },
        set(),
    )
    (end,) = events
    assert isinstance(end, events_.StreamEnd)
    assert end.finish_reason == "stop"
    assert end.provider_metadata == {"gateway": {"finish_reason": "end_turn"}}


def test_finish_object_reason_without_raw() -> None:
    events = v4._parse_stream_part(
        {"type": "finish", "finishReason": {"unified": "tool-calls"}},
        set(),
    )
    (end,) = events
    assert isinstance(end, events_.StreamEnd)
    assert end.finish_reason == "tool_call"
    assert end.provider_metadata is None


def test_response_metadata_populates_stream_end() -> None:
    response_metadata: dict[str, Any] = {}
    assert (
        v4._parse_stream_part(
            {
                "type": "response-metadata",
                "id": "resp-1",
                "modelId": "anthropic/claude-sonnet-4",
            },
            set(),
            None,
            response_metadata,
        )
        == []
    )
    events = v4._parse_stream_part(
        {"type": "finish", "finishReason": "stop"},
        set(),
        None,
        response_metadata,
    )
    (end,) = events
    assert isinstance(end, events_.StreamEnd)
    assert end.response_id == "resp-1"
    assert end.response_model == "anthropic/claude-sonnet-4"


def test_file_part_tagged_data() -> None:
    """v4 wraps generated file data in a tagged union."""
    (event,) = v4._parse_stream_part(
        {
            "type": "file",
            "mediaType": "image/png",
            "data": {"type": "data", "data": "aGk="},
        },
        set(),
    )
    assert isinstance(event, events_.FileEvent)
    assert event.data == "aGk="

    (event,) = v4._parse_stream_part(
        {
            "type": "file",
            "mediaType": "image/png",
            "data": {"type": "url", "url": "https://example.com/a.png"},
        },
        set(),
    )
    assert isinstance(event, events_.FileEvent)
    assert event.data == "https://example.com/a.png"


def test_error_part_raises() -> None:
    with pytest.raises(client_errors.GatewayResponseError):
        v4._parse_stream_part(
            {"type": "error", "error": "boom"},
            set(),
        )


def test_stream_start_warnings_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    warning = {"type": "deprecated", "setting": "topK", "message": "gone"}
    with caplog.at_level("WARNING", logger=v4.logger.name):
        events = v4._parse_stream_part(
            {"type": "stream-start", "warnings": [warning]},
            set(),
        )
    assert events == []
    assert str(warning) in caplog.text


# ---------------------------------------------------------------------------
# _tool_result_output
# ---------------------------------------------------------------------------


def test_tool_result_file_content() -> None:
    """v4 collapses the inline image/file variants into one ``file`` type."""
    fp = messages.FilePart(
        data=b"\x89PNG", media_type="image/png", filename="out.png"
    )
    result = v4._tool_result_output(
        messages.ToolResultPart(
            tool_call_id="tc-1",
            tool_name="t",
            result=messages.ContentOutput(
                value=[messages.TextPart(text="desc"), fp]
            ),
            result_kind="special",
        ),
    )
    assert result["type"] == "content"
    assert result["value"][0] == {"type": "text", "text": "desc"}
    assert result["value"][1] == {
        "type": "file",
        "mediaType": "image/png",
        "data": {
            "type": "data",
            "data": base64.b64encode(b"\x89PNG").decode(),
        },
        "filename": "out.png",
    }
