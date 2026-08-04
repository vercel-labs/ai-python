"""Tests for ``ai.ops.videos`` dispatch behavior.

Gateway wire-format specifics live in
``tests/providers/ai_gateway/test_generate_video.py``; these tests cover
the ops layer: parameter defaults and provider dispatch.
"""

from __future__ import annotations

import ai
from ai import ops

from ..providers.ai_gateway.conftest import user_msg


class TestUnsupportedProvider:
    async def test_generate_video_raises_not_implemented(self) -> None:
        provider = ai.get_provider("openai", api_key="sk-test")
        model = ai.Model(id="sora-2", provider=provider)

        try:
            await ops.generate_video(model, [user_msg("a cat")])
        except NotImplementedError as exc:
            assert "generate_video" in str(exc)
        else:
            raise AssertionError("expected NotImplementedError")
