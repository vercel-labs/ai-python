"""Tests for ``ai.ops.images`` dispatch behavior."""

from __future__ import annotations

from typing import Any, Literal

import ai
from ai import models, ops
from ai.types import messages

from .. import conftest


class ImageProvider(models.Provider):
    provider_class_id: Literal["test-image-provider"] = "test-image-provider"
    name: str = "mock-image"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def generate_image(
        self,
        model: models.Model,
        prompt: Any,
        *,
        params: ops.ImageParams,
    ) -> ops.Item[list[messages.FilePart]]:
        part = messages.FilePart(media_type="image/png", data="aGk=")
        return ops.Item(
            value=[part],
            warnings=[ops.Warning(kind="unsupported", feature="seed")],
        )


async def test_generate_image_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="gpt-image-1", provider=provider)

    try:
        await ops.generate_image(model, "a cat")
    except NotImplementedError as exc:
        assert "generate_image" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


async def test_generate_image_span_with_warnings(
    recorder: conftest.Recorder,
) -> None:
    model = models.Model(id="mock-image-model", provider=ImageProvider())
    await ops.generate_image(model, "a sunset")

    (span,) = recorder.ended
    data = span.data
    assert isinstance(data, ai.experimental_telemetry.GenerateImageSpanData)
    assert data.output_count == 1
    assert data.warnings is not None
    assert data.warnings[0]["kind"] == "unsupported"
