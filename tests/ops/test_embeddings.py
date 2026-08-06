"""Tests for ``ai.ops.embeddings`` dispatch behavior.

Gateway wire-format specifics live in
``tests/providers/ai_gateway/protocol/test_shared.py``; these tests cover
the ops layer: parameter defaults and provider dispatch.
"""

from __future__ import annotations

import ai
from ai import ops


async def test_embed_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="sk-test")
    model = ai.Model(id="text-embedding-3-small", provider=provider)

    try:
        await ops.embed(model, ["hello"])
    except NotImplementedError as exc:
        assert "embed" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
