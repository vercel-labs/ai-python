"""Tests for ``ai.ops.images`` dispatch behavior.

Gateway wire-format specifics live in
``tests/providers/ai_gateway/test_generate_image.py``; these tests cover
the ops layer: parameter defaults and provider dispatch.
"""

from __future__ import annotations

import ai
from ai import ops

from ..providers.ai_gateway.conftest import user_msg


class TestUnsupportedProvider:
    async def test_generate_image_raises_not_implemented(self) -> None:
        provider = ai.get_provider("openai", api_key="sk-test")
        model = ai.Model(id="gpt-image-1", provider=provider)

        try:
            await ops.generate_image(model, [user_msg("a cat")])
        except NotImplementedError as exc:
            assert "generate_image" in str(exc)
        else:
            raise AssertionError("expected NotImplementedError")
