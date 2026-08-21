"""Tests for ``ai.ops.reranking`` dispatch behavior."""

from __future__ import annotations

from typing import Any, Literal

import ai
from ai import models, ops

from .. import conftest


class RerankProvider(models.Provider):
    provider_class_id: Literal["test-rerank-provider"] = "test-rerank-provider"
    name: str = "mock-rerank"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def rerank(
        self,
        model: models.Model,
        documents: list[str] | list[dict[str, Any]],
        query: str,
        *,
        params: ops.RerankParams,
    ) -> ops.Item[list[ops.RankedDocument]]:
        return ops.Item(value=[ops.RankedDocument(index=1, score=0.9)])


async def test_rerank_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="rerank-test", provider=provider)

    try:
        await ops.rerank(model, ["hello"], "query")
    except NotImplementedError as exc:
        assert "rerank" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


async def test_rerank_empty_documents_skips_provider(
    recorder: conftest.Recorder,
) -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="rerank-test", provider=provider)

    result = await ops.rerank(model, [], "query")

    assert result.value == []
    assert result.usage is None
    assert recorder.started == []


async def test_rerank_span(recorder: conftest.Recorder) -> None:
    model = models.Model(id="mock-rerank-model", provider=RerankProvider())
    await ops.rerank(
        model,
        ["doc a", "doc b"],
        "query",
        params=ops.RerankParams(top_n=1),
    )

    (span,) = recorder.ended
    data = span.data
    assert isinstance(data, ai.experimental_telemetry.RerankSpanData)
    assert data.input_count == 2
    assert data.top_n == 1
    assert data.output_count == 1
