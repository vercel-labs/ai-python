"""Tests for ``ai.ops.transcriptions`` dispatch behavior."""

from __future__ import annotations

from typing import Literal

import ai
from ai import models, ops
from ai.types import messages

from .. import conftest


class TranscribeProvider(models.Provider):
    provider_class_id: Literal["test-transcribe-provider"] = (
        "test-transcribe-provider"
    )
    name: str = "mock-transcribe"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def transcribe(
        self,
        model: models.Model,
        audio: messages.FilePart | bytes,
        *,
        params: ops.TranscribeParams,
    ) -> ops.Item[ops.Transcription]:
        return ops.Item(
            value=ops.Transcription(text="hello", duration_seconds=1.5)
        )


async def test_transcribe_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="whisper-1", provider=provider)

    try:
        await ops.transcribe(model, b"\x00\x01")
    except NotImplementedError as exc:
        assert "transcribe" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


async def test_transcribe_span(recorder: conftest.Recorder) -> None:
    model = models.Model(
        id="mock-transcribe-model", provider=TranscribeProvider()
    )
    await ops.transcribe(model, b"audio-bytes")

    (span,) = recorder.ended
    data = span.data
    assert isinstance(data, ai.experimental_telemetry.TranscribeSpanData)
    assert data.duration_seconds == 1.5
