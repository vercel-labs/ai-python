"""Tests for ``ai.ops.images`` dispatch behavior.

Gateway wire-format specifics live in
``tests/providers/ai_gateway/test_protocol.py``; these tests cover
the ops layer: parameter defaults and provider dispatch.
"""

from __future__ import annotations

import ai
from ai import ops


async def test_generate_image_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="sk-test")
    model = ai.Model(id="gpt-image-1", provider=provider)

    try:
        await ops.generate_image(model, "a cat")
    except NotImplementedError as exc:
        assert "generate_image" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
