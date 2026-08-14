"""Tests for ``ai.ops.reranking`` dispatch behavior.

Gateway wire-format specifics live in
``tests/providers/ai_gateway/protocol/test_v4.py``; these tests cover
the ops layer: parameter defaults and provider dispatch.
"""

from __future__ import annotations

import ai
from ai import ops


async def test_rerank_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="sk-test")
    model = ai.Model(id="rerank-test", provider=provider)

    try:
        await ops.rerank(model, ["hello"], "query")
    except NotImplementedError as exc:
        assert "rerank" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


async def test_rerank_empty_documents_skips_provider() -> None:
    provider = ai.get_provider("openai", api_key="sk-test")
    model = ai.Model(id="rerank-test", provider=provider)

    result = await ops.rerank(model, [], "query")

    assert result.value == []
    assert result.usage is None
