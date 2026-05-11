"""Integration tests for the AI Gateway v4 image generation adapter.

Every test exercises the real ``generate()`` function with a ``Client``
wired to an ``httpx.MockTransport``, so the full production code path
is covered:

    generate(client, model, messages, ImageParams(...))
      -> extract prompt/images from messages
      -> httpx POST (mock) to /image-model
      -> JSON response parsing
      -> media type detection
      -> return Message with FileParts
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from ai.models.ai_gateway import ai_gateway, errors
from ai.models.ai_gateway.adapter import generate
from ai.models.core.params import ImageParams
from ai.types import messages

from .conftest import mock_client, user_msg

# 1x1 transparent PNG (minimal valid PNG for magic-byte detection)
_PNG_HEADER = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
_PNG_B64 = base64.b64encode(_PNG_HEADER).decode()

# 1x1 JPEG header
_JPEG_HEADER = bytes([0xFF, 0xD8, 0xFF, 0xE0])
_JPEG_B64 = base64.b64encode(_JPEG_HEADER).decode()

_IMAGE_MODEL = ai_gateway("google/imagen-4.0-generate-001")


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestGenerate:
    async def test_basic_image_generation(self) -> None:
        """Simple prompt -> one PNG image back."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"images": [_PNG_B64]},
            )

        client = mock_client(httpx.MockTransport(handler))
        msg = await generate(
            client, _IMAGE_MODEL, [user_msg("A sunset over Tokyo")], ImageParams()
        )

        assert msg.role == "assistant"
        assert len(msg.images) == 1
        assert msg.images[0].data == _PNG_B64
        assert msg.images[0].media_type == "image/png"

    async def test_multiple_images(self) -> None:
        """Request n=3 images."""

        def handler(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            assert body["n"] == 3
            return httpx.Response(
                200,
                json={"images": [_PNG_B64, _JPEG_B64, _PNG_B64]},
            )

        client = mock_client(httpx.MockTransport(handler))
        msg = await generate(
            client,
            _IMAGE_MODEL,
            [user_msg("Three cats")],
            params=ImageParams(n=3),
        )

        assert len(msg.images) == 3
        assert msg.images[0].media_type == "image/png"
        assert msg.images[1].media_type == "image/jpeg"
        assert msg.images[2].media_type == "image/png"

    async def test_usage_parsing(self) -> None:
        """Usage data from response surfaces on the Message."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "images": [_PNG_B64],
                    "usage": {"inputTokens": 50, "outputTokens": 100},
                },
            )

        client = mock_client(httpx.MockTransport(handler))
        msg = await generate(client, _IMAGE_MODEL, [user_msg("a dog")], ImageParams())

        assert msg.usage is not None
        assert msg.usage.input_tokens == 50
        assert msg.usage.output_tokens == 100


# ---------------------------------------------------------------------------
# Request format
# ---------------------------------------------------------------------------


class TestRequest:
    async def test_protocol_headers(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured.update(dict(req.headers))
            return httpx.Response(200, json={"images": [_PNG_B64]})

        model = ai_gateway("openai/gpt-image-1")
        client = mock_client(httpx.MockTransport(handler), api_key="sk-test")
        await generate(client, model, [user_msg("Hi")], ImageParams())

        assert captured["authorization"] == "Bearer sk-test"
        assert captured["ai-image-model-specification-version"] == "4"
        assert captured["ai-model-id"] == "openai/gpt-image-1"
        assert captured["ai-gateway-auth-method"] == "api-key"

    async def test_request_body_forwards_parameters_and_files(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(200, json={"images": [_PNG_B64]})

        msg = messages.Message(
            role="user",
            parts=[
                messages.TextPart(text="landscape"),
                messages.FilePart(data=_PNG_B64, media_type="image/png"),
            ],
        )
        await generate(
            mock_client(httpx.MockTransport(handler)),
            _IMAGE_MODEL,
            [msg],
            params=ImageParams(
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

    async def test_url_posts_to_image_model_endpoint(self) -> None:
        captured_url: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured_url.append(str(req.url))
            return httpx.Response(200, json={"images": [_PNG_B64]})

        client = mock_client(httpx.MockTransport(handler))
        await generate(client, _IMAGE_MODEL, [user_msg("test")], ImageParams())

        assert captured_url[0] == "https://gw.test/v4/ai/image-model"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_401_authentication_error(self) -> None:
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

        with pytest.raises(errors.GatewayAuthenticationError):
            await generate(
                mock_client(httpx.MockTransport(handler)),
                _IMAGE_MODEL,
                [user_msg("test")],
                ImageParams(),
            )

    async def test_429_rate_limit_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limited",
                        "type": "rate_limit_exceeded",
                    }
                },
            )

        with pytest.raises(errors.GatewayRateLimitError):
            await generate(
                mock_client(httpx.MockTransport(handler)),
                _IMAGE_MODEL,
                [user_msg("test")],
                ImageParams(),
            )

    async def test_empty_images_returns_empty_message(self) -> None:
        """Gateway returns empty images array -> message with no parts."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"images": []})

        msg = await generate(
            mock_client(httpx.MockTransport(handler)),
            _IMAGE_MODEL,
            [user_msg("test")],
            ImageParams(),
        )
        assert len(msg.images) == 0
