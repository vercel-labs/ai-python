"""Tests for ``ai.ops.videos`` dispatch behavior."""

from __future__ import annotations

from typing import Any, Literal

import pytest

import ai
from ai import models, ops
from ai.types import messages

from .. import conftest


class VideoProvider(models.Provider):
    provider_class_id: Literal["test-video-provider"] = "test-video-provider"
    name: str = "mock-video"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def generate_video(
        self,
        model: models.Model,
        prompt: Any,
        *,
        params: ops.VideoParams,
    ) -> ops.Item[list[messages.FilePart]]:
        raise NotImplementedError("mock-video does not support generate_video")


async def test_generate_video_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="sora-2", provider=provider)

    try:
        await ops.generate_video(model, "a cat")
    except NotImplementedError as exc:
        assert "generate_video" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


async def test_error_recorded_on_span(recorder: conftest.Recorder) -> None:
    model = models.Model(id="mock-video-model", provider=VideoProvider())

    with pytest.raises(NotImplementedError):
        await ops.generate_video(model, "a cat")

    (span,) = recorder.ended
    data = span.data
    assert isinstance(data, ai.experimental_telemetry.GenerateVideoSpanData)
    assert data.output_count is None
    assert span.error is not None
    assert span.error.type == "NotImplementedError"
