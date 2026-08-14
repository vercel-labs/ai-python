"""Tests for the version-independent protocol pieces.

Focus areas:
- ``_shared.response_format``: structured-output serialization
- ``_shared.parse_usage``: the two distinct wire formats
- ``_shared.generate_image``/``generate_video``/``generate_audio``: the
  media endpoints, exercised end-to-end via ``ai.ops`` with a provider
  wired to an ``httpx.MockTransport`` (through the default v4 protocol)
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pydantic
import pytest

import ai
from ai import ops
from ai.providers.ai_gateway.protocol import _shared
from ai.types import messages

from ..conftest import mock_model, sse

# ---------------------------------------------------------------------------
# response_format
# ---------------------------------------------------------------------------


def test_response_format_with_output_type() -> None:
    class WeatherResult(pydantic.BaseModel):
        temp: float
        condition: str

    rf = _shared.response_format(WeatherResult)

    assert rf is not None
    assert rf["type"] == "json"
    assert rf["name"] == "WeatherResult"
    assert "properties" in rf["schema"]
    assert "temp" in rf["schema"]["properties"]


def test_response_format_without_output_type() -> None:
    assert _shared.response_format(None) is None


# ---------------------------------------------------------------------------
# _parse_usage
# ---------------------------------------------------------------------------


def test_parse_usage_flat_format() -> None:
    usage = _shared.parse_usage({"prompt_tokens": 10, "completion_tokens": 20})
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20


def test_parse_usage_v3_nested_format() -> None:
    usage = _shared.parse_usage(
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


def test_parse_usage_non_dict_returns_empty() -> None:
    usage = _shared.parse_usage("not a dict")
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


# ---------------------------------------------------------------------------
# generate_image (image-model endpoint, via ops.generate_image)
# ---------------------------------------------------------------------------

# 1x1 transparent PNG (minimal valid PNG for magic-byte detection)
_PNG_HEADER = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
_PNG_B64 = base64.b64encode(_PNG_HEADER).decode()

# 1x1 JPEG header
_JPEG_HEADER = bytes([0xFF, 0xD8, 0xFF, 0xE0])
_JPEG_B64 = base64.b64encode(_JPEG_HEADER).decode()

_IMAGE_MODEL_ID = "google/imagen-4.0-generate-001"


async def test_basic_image_generation() -> None:
    """Simple prompt -> one PNG image back."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"images": [_PNG_B64]},
        )

    model = mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID)
    result = await ops.generate_image(model, "A sunset over Tokyo")
    assert len(result.value) == 1
    assert result.value[0].data == _PNG_B64
    assert result.value[0].media_type == "image/png"


async def test_multiple_images() -> None:
    """Request n=3 images."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["n"] == 3
        return httpx.Response(
            200,
            json={"images": [_PNG_B64, _JPEG_B64, _PNG_B64]},
        )

    result = await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        "Three cats",
        params=ops.ImageParams(n=3),
    )

    assert len(result.value) == 3
    assert result.value[0].media_type == "image/png"
    assert result.value[1].media_type == "image/jpeg"
    assert result.value[2].media_type == "image/png"


async def test_image_usage_parsing() -> None:
    """Usage data from response surfaces on the Item."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "images": [_PNG_B64],
                "usage": {"inputTokens": 50, "outputTokens": 100},
            },
        )

    result = await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        "a dog",
    )

    assert result.usage is not None
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 100


async def test_image_provider_metadata_passthrough() -> None:
    """``providerMetadata`` from the response lands on the item."""
    metadata = {"gateway": {"cost": "0.05", "generationId": "gen-123"}}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"images": [_PNG_B64], "providerMetadata": metadata},
        )

    result = await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        "a dog",
    )

    assert result.provider_metadata == metadata


async def test_image_protocol_headers() -> None:
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        return httpx.Response(200, json={"images": [_PNG_B64]})

    model = mock_model(
        httpx.MockTransport(handler),
        api_key="sk-test",
        model_id="openai/gpt-image-1",
    )
    await ops.generate_image(model, "Hi")

    assert captured["authorization"] == "Bearer sk-test"
    assert captured["ai-image-model-specification-version"] == "4"
    assert captured["ai-model-id"] == "openai/gpt-image-1"
    assert captured["ai-gateway-auth-method"] == "api-key"


async def test_image_request_body_forwards_parameters_and_files() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"images": [_PNG_B64]})

    await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        ops.ImagePrompt(
            text="landscape",
            images=[messages.FilePart(data=_PNG_B64, media_type="image/png")],
        ),
        params=ops.ImageParams(
            n=2,
            size="1024x1024",
            aspect_ratio="16:9",
            seed=42,
            provider_options={"google": {"style": "vivid"}},
        ),
    )

    assert captured_body["prompt"] == "landscape"
    assert captured_body["n"] == 2
    assert captured_body["size"] == "1024x1024"
    assert captured_body["aspectRatio"] == "16:9"
    assert captured_body["seed"] == 42
    assert captured_body["providerOptions"] == {"google": {"style": "vivid"}}
    assert "files" in captured_body
    assert len(captured_body["files"]) == 1
    assert captured_body["files"][0]["type"] == "file"
    assert captured_body["files"][0]["mediaType"] == "image/png"


async def test_image_url_file_part_sent_as_url() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"images": [_PNG_B64]})

    await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        ops.ImagePrompt(
            text="make it watercolor",
            images=[
                messages.FilePart(
                    data="https://example.com/photo.jpg",
                    media_type="image/jpeg",
                )
            ],
        ),
    )

    assert captured_body["files"] == [
        {
            "type": "url",
            "url": "https://example.com/photo.jpg",
            "mediaType": "image/jpeg",
        }
    ]


async def test_image_defaults_omit_unset_parameters() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"images": [_PNG_B64]})

    await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        "test",
    )

    assert captured_body == {"prompt": "test", "n": 1}


async def test_url_posts_to_image_model_endpoint() -> None:
    captured_url: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_url.append(str(req.url))
        return httpx.Response(200, json={"images": [_PNG_B64]})

    await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        "test",
    )

    assert captured_url[0] == "https://gw.test/v4/ai/image-model"


async def test_image_401_authentication_error() -> None:
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
        await ops.generate_image(
            mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
            "test",
        )


async def test_empty_images_returns_empty_item() -> None:
    """Gateway returns empty images array -> item with empty value."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"images": []})

    result = await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        "test",
    )
    assert len(result.value) == 0


async def test_image_mask_and_raw_file_inputs() -> None:
    """bytes / base64 / URL inputs normalize to the wire file format."""
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"images": [_PNG_B64]})

    await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        ops.ImagePrompt(
            text="inpaint the sky",
            images=[_PNG_HEADER, "https://example.com/photo.jpg"],
            mask=_JPEG_B64,
        ),
    )

    assert captured_body["files"] == [
        {"type": "file", "data": _PNG_B64, "mediaType": "image/png"},
        {"type": "url", "url": "https://example.com/photo.jpg"},
    ]
    assert captured_body["mask"] == {
        "type": "file",
        "data": _JPEG_B64,
        "mediaType": "image/jpeg",
    }


async def test_image_data_url_input_decoded_inline() -> None:
    """``data:`` URLs are decoded into inline file data."""
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"images": [_PNG_B64]})

    await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        ops.ImagePrompt(
            images=[f"data:image/webp;base64,{_PNG_B64}"],
        ),
    )

    assert captured_body["files"] == [
        {"type": "file", "data": _PNG_B64, "mediaType": "image/webp"}
    ]
    assert "prompt" not in captured_body


async def test_image_warnings_surface_on_item() -> None:
    """Wire warnings are parsed onto ``Item.warnings``."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "images": [_PNG_B64],
                "warnings": [
                    {
                        "type": "unsupported",
                        "feature": "aspectRatio",
                        "details": "This model ignores aspectRatio.",
                    },
                    {"type": "other", "message": "something odd"},
                ],
            },
        )

    result = await ops.generate_image(
        mock_model(httpx.MockTransport(handler), model_id=_IMAGE_MODEL_ID),
        "test",
        params=ops.ImageParams(aspect_ratio="16:9"),
    )

    assert result.warnings == [
        ops.Warning(
            kind="unsupported",
            feature="aspectRatio",
            details="This model ignores aspectRatio.",
        ),
        ops.Warning(kind="other", message="something odd"),
    ]


# ---------------------------------------------------------------------------
# generate_video (video-model endpoint, SSE, via ops.generate_video)
# ---------------------------------------------------------------------------

# MP4 magic bytes (ftyp box)
_MP4_HEADER = bytes(
    [0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6F, 0x6D]
)
_MP4_B64 = base64.b64encode(_MP4_HEADER).decode()

# WebM magic bytes
_WEBM_HEADER = bytes([0x1A, 0x45, 0xDF, 0xA3])
_WEBM_B64 = base64.b64encode(_WEBM_HEADER).decode()

_VIDEO_MODEL_ID = "google/veo-3.0-generate-001"

# A single-MP4 "result" SSE body, for tests that don't inspect the videos.
_MP4_RESULT_SSE = sse(
    {
        "type": "result",
        "videos": [
            {"type": "base64", "data": _MP4_B64, "mediaType": "video/mp4"}
        ],
    }
)


async def test_basic_video_generation_base64() -> None:
    """Simple prompt -> one MP4 video back via base64."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    result = await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        "A cat walking on a beach",
    )
    assert len(result.value) == 1
    assert result.value[0].data == _MP4_B64
    assert result.value[0].media_type == "video/mp4"


async def test_video_generation_url() -> None:
    """Video returned as URL -> downloaded automatically."""
    body = sse(
        {
            "type": "result",
            "videos": [
                {
                    "type": "url",
                    "url": "https://storage.example.com/video.mp4",
                    "mediaType": "video/mp4",
                }
            ],
        }
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    model = mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID)

    with patch(
        "ai.models.core.helpers.files.download",
        new_callable=AsyncMock,
        return_value=(_MP4_HEADER, "video/mp4"),
    ) as mock_dl:
        result = await ops.generate_video(
            model,
            "A sunset timelapse",
        )

    mock_dl.assert_called_once_with("https://storage.example.com/video.mp4")
    assert len(result.value) == 1
    assert result.value[0].data == _MP4_HEADER
    assert result.value[0].media_type == "video/mp4"


async def test_multiple_videos() -> None:
    body = sse(
        {
            "type": "result",
            "videos": [
                {
                    "type": "base64",
                    "data": _MP4_B64,
                    "mediaType": "video/mp4",
                },
                {
                    "type": "base64",
                    "data": _WEBM_B64,
                    "mediaType": "video/webm",
                },
            ],
        }
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    result = await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        "Two versions",
        params=ops.VideoParams(n=2),
    )
    assert len(result.value) == 2
    assert result.value[0].media_type == "video/mp4"
    assert result.value[1].media_type == "video/webm"


async def test_video_provider_metadata_passthrough() -> None:
    """``providerMetadata`` from the result event lands on the item."""
    metadata = {
        "gateway": {"cost": "0.20", "generationId": "gen-xyz-789"},
        "fal": {"usage": {"computeUnits": 10}},
    }
    body = sse(
        {
            "type": "result",
            "videos": [
                {
                    "type": "base64",
                    "data": _MP4_B64,
                    "mediaType": "video/mp4",
                }
            ],
            "providerMetadata": metadata,
        }
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    result = await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        "a dog",
    )

    assert result.provider_metadata == metadata
    assert result.usage is None


async def test_video_protocol_headers() -> None:
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    model = mock_model(
        httpx.MockTransport(handler),
        api_key="sk-test",
        model_id=_VIDEO_MODEL_ID,
    )
    await ops.generate_video(model, "test")

    assert captured["authorization"] == "Bearer sk-test"
    assert captured["ai-video-model-specification-version"] == "4"
    assert captured["ai-model-id"] == "google/veo-3.0-generate-001"
    assert captured["accept"] == "text/event-stream"
    assert captured["ai-gateway-auth-method"] == "api-key"


async def test_video_request_body_forwards_parameters_and_input_image() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    png_b64 = base64.b64encode(b"\x89PNG").decode()
    await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        ops.VideoPrompt(
            text="sunset",
            image=messages.FilePart(data=png_b64, media_type="image/png"),
        ),
        params=ops.VideoParams(
            n=2,
            aspect_ratio="16:9",
            resolution="1920x1080",
            duration=5,
            fps=30,
            seed=42,
            generate_audio=True,
            provider_options={"google": {"enhancePrompt": True}},
        ),
    )

    assert captured_body["prompt"] == "sunset"
    assert captured_body["n"] == 2
    assert captured_body["aspectRatio"] == "16:9"
    assert captured_body["resolution"] == "1920x1080"
    assert captured_body["duration"] == 5
    assert captured_body["fps"] == 30
    assert captured_body["seed"] == 42
    assert captured_body["generateAudio"] is True
    assert captured_body["providerOptions"] == {
        "google": {"enhancePrompt": True}
    }
    assert "image" in captured_body
    assert captured_body["image"]["type"] == "file"
    assert captured_body["image"]["mediaType"] == "image/png"


async def test_video_defaults_omit_unset_parameters() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        "test",
    )

    assert captured_body == {"prompt": "test", "n": 1}


async def test_url_posts_to_video_model_endpoint() -> None:
    captured_url: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_url.append(str(req.url))
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        "test",
    )
    assert captured_url[0] == "https://gw.test/v4/ai/video-model"


async def test_video_sse_error_event() -> None:
    """Gateway returns an SSE error event -> raises."""
    body = sse(
        {
            "type": "error",
            "message": "Content policy violation",
            "errorType": "content_filter",
            "statusCode": 400,
            "param": None,
        }
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(ai.ProviderBadRequestError, match="Content policy"):
        await ops.generate_video(
            mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
            "test",
        )


async def test_video_401_authentication_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Bad key",
                    "type": "authentication_error",
                }
            },
        )

    with pytest.raises(ai.ProviderAuthenticationError):
        await ops.generate_video(
            mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
            "test",
        )


async def test_video_empty_sse_stream() -> None:
    """SSE stream with no data events -> raises."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    with pytest.raises(ai.ProviderResponseError, match="SSE stream ended"):
        await ops.generate_video(
            mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
            "test",
        )


async def test_video_frame_images_wire_format() -> None:
    """Role-tagged frame images land in ``frameImages``; the first_frame
    entry doubles as the start ``image``."""
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        ops.VideoPrompt(
            text="morph",
            frame_images=[
                ops.FrameImage(image=_PNG_HEADER, frame_type="first_frame"),
                ops.FrameImage(
                    image="https://example.com/end.png",
                    frame_type="last_frame",
                ),
            ],
        ),
    )

    assert captured_body["frameImages"] == [
        {
            "image": {
                "type": "file",
                "data": _PNG_B64,
                "mediaType": "image/png",
            },
            "frameType": "first_frame",
        },
        {
            "image": {"type": "url", "url": "https://example.com/end.png"},
            "frameType": "last_frame",
        },
    ]
    assert captured_body["image"] == {
        "type": "file",
        "data": _PNG_B64,
        "mediaType": "image/png",
    }


async def test_video_references_wire_format() -> None:
    """References route by media type: images and videos both pass."""
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        ops.VideoPrompt(
            text="in this style",
            references=[
                _MP4_HEADER,
                messages.FilePart(
                    data="https://example.com/ref.mp4",
                    media_type="video/mp4",
                ),
                _PNG_HEADER,
            ],
        ),
    )

    assert captured_body["inputReferences"] == [
        {"type": "file", "data": _MP4_B64, "mediaType": "video/mp4"},
        {
            "type": "url",
            "url": "https://example.com/ref.mp4",
            "mediaType": "video/mp4",
        },
        {"type": "file", "data": _PNG_B64, "mediaType": "image/png"},
    ]
    assert "image" not in captured_body
    assert "frameImages" not in captured_body


async def test_video_frame_images_suppress_references_and_image() -> None:
    """frame_images win over references and the start image, with
    warnings surfaced on the item."""
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, text=_MP4_RESULT_SSE)

    result = await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        ops.VideoPrompt(
            text="morph",
            image=_JPEG_HEADER,
            frame_images=[
                ops.FrameImage(image=_PNG_HEADER, frame_type="first_frame"),
            ],
            references=[_MP4_HEADER],
        ),
    )

    assert "inputReferences" not in captured_body
    assert captured_body["image"] == {
        "type": "file",
        "data": _PNG_B64,
        "mediaType": "image/png",
    }
    assert [w.kind for w in result.warnings] == ["other", "other"]
    assert "references were ignored" in (result.warnings[0].message or "")
    assert "image was ignored" in (result.warnings[1].message or "")


async def test_video_warnings_surface_on_item() -> None:
    """Wire warnings from the result event land on ``Item.warnings``."""
    body = sse(
        {
            "type": "result",
            "videos": [
                {
                    "type": "base64",
                    "data": _MP4_B64,
                    "mediaType": "video/mp4",
                }
            ],
            "warnings": [{"type": "unsupported", "feature": "fps"}],
        }
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    result = await ops.generate_video(
        mock_model(httpx.MockTransport(handler), model_id=_VIDEO_MODEL_ID),
        "test",
        params=ops.VideoParams(fps=60),
    )

    assert result.warnings == [ops.Warning(kind="unsupported", feature="fps")]


# ---------------------------------------------------------------------------
# generate_audio (speech-model endpoint, via ops.generate_audio)
# ---------------------------------------------------------------------------

# MP3 frame sync header (minimal valid MP3 for magic-byte detection)
_MP3_HEADER = bytes([0xFF, 0xFB, 0x90, 0x64])
_MP3_B64 = base64.b64encode(_MP3_HEADER).decode()

# RIFF....WAVE header
_WAV_HEADER = bytes([0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00]) + b"WAVE"
_WAV_B64 = base64.b64encode(_WAV_HEADER).decode()

_SPEECH_MODEL_ID = "openai/tts-1"


async def test_basic_speech_generation() -> None:
    """Simple text -> one MP3 audio file back."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"audio": _MP3_B64},
        )

    model = mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID)
    result = await ops.generate_audio(model, "Hello world")
    assert len(result.value) == 1
    assert result.value[0].data == _MP3_B64
    assert result.value[0].media_type == "audio/mpeg"


async def test_wav_media_type_detection() -> None:
    """WAV magic bytes are detected as audio/wav."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"audio": _WAV_B64})

    result = await ops.generate_audio(
        mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
        "Hello world",
        params=ops.AudioParams(output_format="wav"),
    )

    assert result.value[0].media_type == "audio/wav"


async def test_unknown_bytes_fall_back_to_mpeg() -> None:
    """Undetectable audio data defaults to audio/mpeg."""
    unknown_b64 = base64.b64encode(b"\x01\x02\x03\x04").decode()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"audio": unknown_b64})

    result = await ops.generate_audio(
        mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
        "Hello world",
    )

    assert result.value[0].media_type == "audio/mpeg"


async def test_audio_provider_metadata_passthrough() -> None:
    """``providerMetadata`` from the response lands on the item."""
    metadata = {"gateway": {"cost": "0.05", "generationId": "gen-123"}}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"audio": _MP3_B64, "providerMetadata": metadata},
        )

    result = await ops.generate_audio(
        mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
        "Hello world",
    )

    assert result.provider_metadata == metadata


async def test_audio_protocol_headers() -> None:
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        return httpx.Response(200, json={"audio": _MP3_B64})

    model = mock_model(
        httpx.MockTransport(handler),
        api_key="sk-test",
        model_id=_SPEECH_MODEL_ID,
    )
    await ops.generate_audio(model, "Hi")

    assert captured["authorization"] == "Bearer sk-test"
    assert captured["ai-speech-model-specification-version"] == "4"
    assert captured["ai-model-id"] == _SPEECH_MODEL_ID
    assert captured["ai-gateway-auth-method"] == "api-key"


async def test_audio_request_body_forwards_parameters() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"audio": _MP3_B64})

    await ops.generate_audio(
        mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
        ops.AudioPrompt(text="Hello world", instructions="Speak slowly"),
        params=ops.AudioParams(
            voice="alloy",
            output_format="wav",
            speed=1.5,
            language="en",
            provider_options={"openai": {"style": "calm"}},
        ),
    )

    assert captured_body["text"] == "Hello world"
    assert captured_body["voice"] == "alloy"
    assert captured_body["outputFormat"] == "wav"
    assert captured_body["instructions"] == "Speak slowly"
    assert captured_body["speed"] == 1.5
    assert captured_body["language"] == "en"
    assert captured_body["providerOptions"] == {"openai": {"style": "calm"}}


async def test_audio_defaults_omit_unset_parameters() -> None:
    captured_body: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(req.content))
        return httpx.Response(200, json={"audio": _MP3_B64})

    await ops.generate_audio(
        mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
        "test",
    )

    assert captured_body == {"text": "test"}


async def test_url_posts_to_speech_model_endpoint() -> None:
    captured_url: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_url.append(str(req.url))
        return httpx.Response(200, json={"audio": _MP3_B64})

    await ops.generate_audio(
        mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
        "test",
    )

    assert captured_url[0] == "https://gw.test/v4/ai/speech-model"


async def test_audio_401_authentication_error() -> None:
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
        await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            "test",
        )


async def test_missing_audio_returns_empty_item() -> None:
    """Gateway returns no audio -> item with empty value."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"audio": ""})

    result = await ops.generate_audio(
        mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
        "test",
    )
    assert len(result.value) == 0
