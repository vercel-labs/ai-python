"""Tests for ``ai.ops.embeddings`` dispatch behavior."""

from __future__ import annotations

from typing import Literal

import ai
from ai import models, ops
from ai.types import usage

from .. import conftest


class EmbedProvider(models.Provider):
    provider_class_id: Literal["test-embed-provider"] = "test-embed-provider"
    name: str = "mock-embed"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def embed(
        self,
        model: models.Model,
        values: list[str],
        *,
        params: ops.EmbedParams,
    ) -> ops.Item[list[list[float]]]:
        return ops.Item(
            value=[[0.1, 0.2, 0.3] for _ in values],
            usage=usage.Usage(input_tokens=7),
        )


async def test_embed_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="text-embedding-3-small", provider=provider)

    try:
        await ops.embed(model, ["hello"])
    except NotImplementedError as exc:
        assert "embed" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


async def test_embed_span(recorder: conftest.Recorder) -> None:
    model = models.Model(id="mock-embed-model", provider=EmbedProvider())
    await ops.embed(model, ["hello", "world"])

    (span,) = recorder.ended
    data = span.data
    assert isinstance(data, ai.experimental_telemetry.EmbedSpanData)
    assert data.model == "mock-embed-model"
    assert data.provider == "mock-embed"
    assert data.input_count == 2
    assert data.output_count == 2
    assert data.dimensions == 3
    assert data.usage is not None and data.usage.input_tokens == 7
    assert span.started_at is not None
    assert span.ended_at is not None
    assert span.error is None

    (started,) = recorder.started
    assert isinstance(started.data, ai.experimental_telemetry.EmbedSpanData)
