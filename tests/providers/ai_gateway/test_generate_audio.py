"""Integration tests for the AI Gateway v3 speech generation adapter.

Every test exercises the real ``ops.generate_audio()`` function with a
provider wired to an ``httpx.MockTransport``, so the full production
code path is covered:

    ops.generate_audio(model, messages, params=AudioParams(...))
      -> extract text from messages
      -> httpx POST (mock) to /speech-model
      -> JSON response parsing
      -> media type detection
      -> return Message with a FilePart
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

import ai
from ai import ops

from .conftest import mock_model, user_msg

# MP3 frame sync header (minimal valid MP3 for magic-byte detection)
_MP3_HEADER = bytes([0xFF, 0xFB, 0x90, 0x64])
_MP3_B64 = base64.b64encode(_MP3_HEADER).decode()

# RIFF....WAVE header
_WAV_HEADER = bytes([0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00]) + b"WAVE"
_WAV_B64 = base64.b64encode(_WAV_HEADER).decode()

_SPEECH_MODEL_ID = "openai/tts-1"


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestGenerate:
    async def test_basic_speech_generation(self) -> None:
        """Simple text -> one MP3 audio file back."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"audio": _MP3_B64},
            )

        model = mock_model(
            httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID
        )
        msg = await ops.generate_audio(model, [user_msg("Hello world")])

        assert msg.role == "assistant"
        assert len(msg.audio) == 1
        assert msg.audio[0].data == _MP3_B64
        assert msg.audio[0].media_type == "audio/mpeg"

    async def test_wav_media_type_detection(self) -> None:
        """WAV magic bytes are detected as audio/wav."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"audio": _WAV_B64})

        msg = await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            [user_msg("Hello world")],
            params=ops.AudioParams(output_format="wav"),
        )

        assert msg.audio[0].media_type == "audio/wav"

    async def test_unknown_bytes_fall_back_to_mpeg(self) -> None:
        """Undetectable audio data defaults to audio/mpeg."""
        unknown_b64 = base64.b64encode(b"\x01\x02\x03\x04").decode()

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"audio": unknown_b64})

        msg = await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            [user_msg("Hello world")],
        )

        assert msg.audio[0].media_type == "audio/mpeg"

    async def test_provider_metadata_passthrough(self) -> None:
        """``providerMetadata`` from the response lands on the message."""
        metadata = {"gateway": {"cost": "0.05", "generationId": "gen-123"}}

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"audio": _MP3_B64, "providerMetadata": metadata},
            )

        msg = await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            [user_msg("Hello world")],
        )

        assert msg.provider_metadata == metadata


# ---------------------------------------------------------------------------
# Request format
# ---------------------------------------------------------------------------


class TestRequest:
    async def test_protocol_headers(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured.update(dict(req.headers))
            return httpx.Response(200, json={"audio": _MP3_B64})

        model = mock_model(
            httpx.MockTransport(handler),
            api_key="sk-test",
            model_id=_SPEECH_MODEL_ID,
        )
        await ops.generate_audio(model, [user_msg("Hi")])

        assert captured["authorization"] == "Bearer sk-test"
        assert captured["ai-speech-model-specification-version"] == "3"
        assert captured["ai-model-id"] == _SPEECH_MODEL_ID
        assert captured["ai-gateway-auth-method"] == "api-key"

    async def test_request_body_forwards_parameters(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(200, json={"audio": _MP3_B64})

        await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            [user_msg("Hello world")],
            params=ops.AudioParams(
                voice="alloy",
                output_format="wav",
                instructions="Speak slowly",
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

    async def test_defaults_omit_unset_parameters(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(200, json={"audio": _MP3_B64})

        await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            [user_msg("test")],
        )

        assert captured_body == {"text": "test"}

    async def test_url_posts_to_speech_model_endpoint(self) -> None:
        captured_url: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured_url.append(str(req.url))
            return httpx.Response(200, json={"audio": _MP3_B64})

        await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            [user_msg("test")],
        )

        assert captured_url[0] == "https://gw.test/v3/ai/speech-model"


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

        with pytest.raises(ai.ProviderAuthenticationError):
            await ops.generate_audio(
                mock_model(
                    httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID
                ),
                [user_msg("test")],
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

        with pytest.raises(ai.ProviderRateLimitError):
            await ops.generate_audio(
                mock_model(
                    httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID
                ),
                [user_msg("test")],
            )

    async def test_missing_audio_returns_empty_message(self) -> None:
        """Gateway returns no audio -> message with no parts."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"audio": ""})

        msg = await ops.generate_audio(
            mock_model(httpx.MockTransport(handler), model_id=_SPEECH_MODEL_ID),
            [user_msg("test")],
        )
        assert len(msg.audio) == 0
