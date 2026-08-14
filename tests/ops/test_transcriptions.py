"""Tests for ``ai.ops.transcriptions`` dispatch behavior.

Gateway wire-format specifics live in
``tests/providers/ai_gateway/protocol/test_v4.py``; these tests cover
the ops layer: parameter defaults and provider dispatch.
"""

from __future__ import annotations

import ai
from ai import ops


async def test_transcribe_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="sk-test")
    model = ai.Model(id="whisper-1", provider=provider)

    try:
        await ops.transcribe(model, b"\x00\x01")
    except NotImplementedError as exc:
        assert "transcribe" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
