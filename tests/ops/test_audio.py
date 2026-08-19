"""Tests for ``ai.ops.audio`` dispatch behavior."""

from __future__ import annotations

from typing import Literal

import ai
from ai import models, ops
from ai.types import messages

from .. import conftest


class AudioProvider(models.Provider):
    provider_class_id: Literal["test-audio-provider"] = "test-audio-provider"
    name: str = "mock-audio"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def generate_audio(
        self,
        model: models.Model,
        prompt: ops.AudioPrompt,
        *,
        params: ops.AudioParams,
    ) -> ops.Item[list[messages.FilePart]]:
        part = messages.FilePart(media_type="audio/mpeg", data="aGk=")
        return ops.Item(value=[part])


async def test_generate_audio_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="tts-1", provider=provider)

    try:
        await ops.generate_audio(model, "Hello!")
    except NotImplementedError as exc:
        assert "generate_audio" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


async def test_generate_audio_span(recorder: conftest.Recorder) -> None:
    model = models.Model(id="mock-audio-model", provider=AudioProvider())
    await ops.generate_audio(model, "Hello!")

    (span,) = recorder.ended
    data = span.data
    assert isinstance(data, ai.experimental_telemetry.GenerateAudioSpanData)
    assert data.model == "mock-audio-model"
    assert data.provider == "mock-audio"
    assert data.output_count == 1
