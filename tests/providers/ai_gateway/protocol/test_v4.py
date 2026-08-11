"""Tests for the v4 protocol serialization and deserialization.

Focus areas:
- ``v4._messages_to_prompt``: tagged file-data unions, URL passthrough
- ``v4._apply_reasoning``: the standardized effort field
- ``v4._parse_stream_part``: object finish reasons, tagged file data,
  response metadata, error parts, warnings
- ``v4.GatewayV4Protocol.embed``: the v4-only embedding endpoint,
  exercised end-to-end via ``ai.ops.embed`` with a provider wired to
  an ``httpx.MockTransport``
- ``v4.GatewayV4Protocol.transcribe``: the v4-only transcription
  endpoint, exercised end-to-end via ``ai.ops.transcribe``
- ``v4.GatewayV4Protocol.rerank``: the v4-only reranking endpoint,
  exercised end-to-end via ``ai.ops.rerank``

Version-independent pieces are tested in ``test_shared.py``; end-to-end
request/response behavior in ``../test_stream.py``.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

import ai
from ai import models, ops
from ai.providers.ai_gateway.client import errors as client_errors
from ai.providers.ai_gateway.protocol import v4
from ai.types import events as events_
from ai.types import messages

from ..conftest import mock_model

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


# ---------------------------------------------------------------------------
# embed (embedding-model endpoint, via ops.embed)
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL_ID = "openai/text-embedding-3-small"
_EMBEDDINGS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


async def test_basic_embed() -> None:
    """Two strings in -> two vectors back, with usage."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": _EMBEDDINGS, "usage": {"tokens": 4}},
        )

    model = mock_model(
        httpx.MockTransport(handler), model_id=_EMBEDDING_MODEL_ID
    )
    result = await ops.embed(model, ["hello", "world"])

    assert result.value == _EMBEDDINGS
    assert result.usage is not None
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 0


async def test_embed_request_body_and_url() -> None:
    captured_body: dict[str, Any] = {}
    captured_url: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        captured_url.append(str(req.url))
        return httpx.Response(200, json={"embeddings": _EMBEDDINGS})

    await ops.embed(
        mock_model(httpx.MockTransport(handler), model_id=_EMBEDDING_MODEL_ID),
        ["hello", "world"],
    )

    assert captured_body == {"values": ["hello", "world"]}
    assert captured_url[0] == "https://gw.test/v4/ai/embedding-model"


async def test_embed_protocol_headers() -> None:
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        return httpx.Response(200, json={"embeddings": _EMBEDDINGS})

    model = mock_model(
        httpx.MockTransport(handler),
        api_key="sk-test",
        model_id=_EMBEDDING_MODEL_ID,
    )
    await ops.embed(model, ["hi"])

    assert captured["authorization"] == "Bearer sk-test"
    assert captured["ai-embedding-model-specification-version"] == "4"
    assert captured["ai-model-id"] == _EMBEDDING_MODEL_ID
    assert captured["ai-gateway-auth-method"] == "api-key"


async def test_embed_forwards_provider_options() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"embeddings": _EMBEDDINGS})

    await ops.embed(
        mock_model(httpx.MockTransport(handler), model_id=_EMBEDDING_MODEL_ID),
        ["hello"],
        params=ops.EmbedParams(
            provider_options={"openai": {"dimensions": 256}}
        ),
    )

    assert captured_body["providerOptions"] == {"openai": {"dimensions": 256}}


async def test_embed_provider_metadata_and_missing_usage() -> None:
    """``providerMetadata`` lands on the item; absent usage stays None."""
    metadata = {"gateway": {"cost": "0.0001"}}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": _EMBEDDINGS, "providerMetadata": metadata},
        )

    result = await ops.embed(
        mock_model(httpx.MockTransport(handler), model_id=_EMBEDDING_MODEL_ID),
        ["hello"],
    )

    assert result.provider_metadata == metadata
    assert result.usage is None


async def test_embed_401_authentication_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
        )

    with pytest.raises(ai.ProviderAuthenticationError):
        await ops.embed(
            mock_model(
                httpx.MockTransport(handler), model_id=_EMBEDDING_MODEL_ID
            ),
            ["test"],
        )


# ---------------------------------------------------------------------------
# transcribe (transcription-model endpoint, via ops.transcribe)
# ---------------------------------------------------------------------------

_TRANSCRIPTION_MODEL_ID = "openai/whisper-1"

# MP3 frame sync header (minimal valid MP3 for magic-byte detection)
_MP3_HEADER = bytes([0xFF, 0xFB, 0x90, 0x64])
_MP3_B64 = base64.b64encode(_MP3_HEADER).decode()


async def test_basic_transcribe() -> None:
    """Raw bytes in -> transcript with segments and metadata back."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "text": "Hello world",
                "segments": [
                    {"text": "Hello", "startSecond": 0.0, "endSecond": 0.4},
                    {"text": "world", "startSecond": 0.4, "endSecond": 0.8},
                ],
                "language": "en",
                "durationInSeconds": 0.8,
            },
        )

    model = mock_model(
        httpx.MockTransport(handler), model_id=_TRANSCRIPTION_MODEL_ID
    )
    result = await ops.transcribe(model, _MP3_HEADER)

    assert result.value.text == "Hello world"
    assert [s.text for s in result.value.segments] == ["Hello", "world"]
    assert result.value.segments[1].start_second == 0.4
    assert result.value.segments[1].end_second == 0.8
    assert result.value.language == "en"
    assert result.value.duration_seconds == 0.8
    assert result.usage is None


async def test_transcribe_request_body_and_url() -> None:
    """Bytes are base64-encoded, media type detected from magic bytes."""
    captured_body: dict[str, Any] = {}
    captured_url: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        captured_url.append(str(req.url))
        return httpx.Response(200, json={"text": "hi"})

    await ops.transcribe(
        mock_model(
            httpx.MockTransport(handler), model_id=_TRANSCRIPTION_MODEL_ID
        ),
        _MP3_HEADER,
    )

    assert captured_body == {"audio": _MP3_B64, "mediaType": "audio/mpeg"}
    assert captured_url[0] == "https://gw.test/v4/ai/transcription-model"


async def test_transcribe_unknown_bytes_fall_back_to_wav() -> None:
    """Undetectable audio bytes default to audio/wav."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["mediaType"] == "audio/wav"
        return httpx.Response(200, json={"text": "hi"})

    await ops.transcribe(
        mock_model(
            httpx.MockTransport(handler), model_id=_TRANSCRIPTION_MODEL_ID
        ),
        b"\x01\x02\x03\x04",
    )


async def test_transcribe_file_part_media_type_wins() -> None:
    """A FilePart's media_type is sent as-is, data passed through as b64."""
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"text": "hi"})

    await ops.transcribe(
        mock_model(
            httpx.MockTransport(handler), model_id=_TRANSCRIPTION_MODEL_ID
        ),
        messages.FilePart(data=_MP3_B64, media_type="audio/mp4"),
    )

    assert captured_body == {"audio": _MP3_B64, "mediaType": "audio/mp4"}


async def test_transcribe_file_part_url_is_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An http(s) FilePart is fetched client-side and sent inline."""
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"text": "hi"})

    async def fake_download(url: str, **_: Any) -> tuple[bytes, str | None]:
        assert url == "https://example.com/speech.mp3"
        return _MP3_HEADER, "audio/mpeg"

    monkeypatch.setattr(models.core.helpers.files, "download", fake_download)

    await ops.transcribe(
        mock_model(
            httpx.MockTransport(handler),
            model_id=_TRANSCRIPTION_MODEL_ID,
        ),
        messages.FilePart(
            data="https://example.com/speech.mp3",
            media_type="audio/mpeg",
        ),
    )

    assert captured_body == {"audio": _MP3_B64, "mediaType": "audio/mpeg"}


async def test_transcribe_protocol_headers() -> None:
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        return httpx.Response(200, json={"text": "hi"})

    model = mock_model(
        httpx.MockTransport(handler),
        api_key="sk-test",
        model_id=_TRANSCRIPTION_MODEL_ID,
    )
    await ops.transcribe(model, _MP3_HEADER)

    assert captured["authorization"] == "Bearer sk-test"
    assert captured["ai-transcription-model-specification-version"] == "4"
    assert captured["ai-model-id"] == _TRANSCRIPTION_MODEL_ID
    assert captured["ai-gateway-auth-method"] == "api-key"


async def test_transcribe_forwards_provider_options() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"text": "hi"})

    await ops.transcribe(
        mock_model(
            httpx.MockTransport(handler), model_id=_TRANSCRIPTION_MODEL_ID
        ),
        _MP3_HEADER,
        params=ops.TranscribeParams(
            provider_options={"openai": {"timestampGranularities": ["word"]}}
        ),
    )

    assert captured_body["providerOptions"] == {
        "openai": {"timestampGranularities": ["word"]}
    }


async def test_transcribe_provider_metadata_passthrough() -> None:
    """``providerMetadata`` from the response lands on the item."""
    metadata = {"gateway": {"cost": "0.006"}}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"text": "hi", "providerMetadata": metadata},
        )

    result = await ops.transcribe(
        mock_model(
            httpx.MockTransport(handler), model_id=_TRANSCRIPTION_MODEL_ID
        ),
        _MP3_HEADER,
    )

    assert result.provider_metadata == metadata


async def test_transcribe_401_authentication_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
        )

    with pytest.raises(ai.ProviderAuthenticationError):
        await ops.transcribe(
            mock_model(
                httpx.MockTransport(handler),
                model_id=_TRANSCRIPTION_MODEL_ID,
            ),
            _MP3_HEADER,
        )


# ---------------------------------------------------------------------------
# rerank (reranking-model endpoint, via ops.rerank)
# ---------------------------------------------------------------------------

_RERANKING_MODEL_ID = "cohere/rerank-v3.5"
_RANKING = [
    {"index": 2, "relevanceScore": 0.9},
    {"index": 0, "relevanceScore": 0.4},
]
_DOCUMENTS = ["apple pie", "stock market", "capital of Japan"]


async def test_basic_rerank() -> None:
    """Documents in -> ranking of index/score pairs back, server order."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ranking": _RANKING})

    model = mock_model(
        httpx.MockTransport(handler), model_id=_RERANKING_MODEL_ID
    )
    result = await ops.rerank(model, _DOCUMENTS, "tokyo")

    assert result.value == [
        ops.RankedDocument(index=2, score=0.9),
        ops.RankedDocument(index=0, score=0.4),
    ]
    assert result.usage is None


async def test_rerank_request_body_and_url() -> None:
    captured_body: dict[str, Any] = {}
    captured_url: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        captured_url.append(str(req.url))
        return httpx.Response(200, json={"ranking": _RANKING})

    await ops.rerank(
        mock_model(httpx.MockTransport(handler), model_id=_RERANKING_MODEL_ID),
        _DOCUMENTS,
        "tokyo",
    )

    assert captured_body == {
        "query": "tokyo",
        "documents": {"type": "text", "values": _DOCUMENTS},
    }
    assert captured_url[0] == "https://gw.test/v4/ai/reranking-model"


async def test_rerank_object_documents() -> None:
    """dict documents are sent with the ``object`` wire type."""
    captured_body: dict[str, Any] = {}
    documents = [{"title": "pie"}, {"title": "tokyo"}]

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"ranking": _RANKING})

    await ops.rerank(
        mock_model(httpx.MockTransport(handler), model_id=_RERANKING_MODEL_ID),
        documents,
        "tokyo",
    )

    assert captured_body["documents"] == {
        "type": "object",
        "values": documents,
    }


async def test_rerank_top_n_only_when_set() -> None:
    captured_bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"ranking": _RANKING})

    model = mock_model(
        httpx.MockTransport(handler), model_id=_RERANKING_MODEL_ID
    )
    await ops.rerank(model, _DOCUMENTS, "tokyo")
    await ops.rerank(
        model, _DOCUMENTS, "tokyo", params=ops.RerankParams(top_n=2)
    )

    assert "topN" not in captured_bodies[0]
    assert captured_bodies[1]["topN"] == 2


async def test_rerank_protocol_headers() -> None:
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        return httpx.Response(200, json={"ranking": _RANKING})

    model = mock_model(
        httpx.MockTransport(handler),
        api_key="sk-test",
        model_id=_RERANKING_MODEL_ID,
    )
    await ops.rerank(model, _DOCUMENTS, "tokyo")

    assert captured["authorization"] == "Bearer sk-test"
    assert captured["ai-reranking-model-specification-version"] == "4"
    assert captured["ai-model-id"] == _RERANKING_MODEL_ID
    assert captured["ai-gateway-auth-method"] == "api-key"


async def test_rerank_forwards_provider_options() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"ranking": _RANKING})

    await ops.rerank(
        mock_model(httpx.MockTransport(handler), model_id=_RERANKING_MODEL_ID),
        _DOCUMENTS,
        "tokyo",
        params=ops.RerankParams(
            provider_options={"cohere": {"maxTokensPerDoc": 512}}
        ),
    )

    assert captured_body["providerOptions"] == {
        "cohere": {"maxTokensPerDoc": 512}
    }


async def test_rerank_provider_metadata_and_warnings() -> None:
    """``providerMetadata`` lands on the item; warnings surface on it."""
    metadata = {"gateway": {"cost": "0.002"}}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ranking": _RANKING,
                "providerMetadata": metadata,
                "warnings": [{"type": "other", "message": "topN clamped"}],
            },
        )

    result = await ops.rerank(
        mock_model(httpx.MockTransport(handler), model_id=_RERANKING_MODEL_ID),
        _DOCUMENTS,
        "tokyo",
    )

    assert result.provider_metadata == metadata
    assert result.warnings == [
        ops.Warning(kind="other", message="topN clamped")
    ]


async def test_rerank_401_authentication_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
        )

    with pytest.raises(ai.ProviderAuthenticationError):
        await ops.rerank(
            mock_model(
                httpx.MockTransport(handler), model_id=_RERANKING_MODEL_ID
            ),
            _DOCUMENTS,
            "tokyo",
        )
