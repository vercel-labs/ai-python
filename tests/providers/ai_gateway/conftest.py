"""Shared helpers for AI Gateway tests."""

from __future__ import annotations

import json
from typing import Any

import httpx

import ai
from ai.types import messages

_BASE_URL = "https://gw.test/v3/ai"


def sse(*events: dict[str, Any]) -> str:
    """Build SSE response text from event dicts."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def mock_model(
    handler: httpx.MockTransport,
    *,
    model_id: str = "test-provider/test-model",
    api_key: str = "test-key",
) -> ai.Model:
    """Create a Gateway model wired to a mock transport.

    Per-test handlers are live objects, so the factory is a closure and
    the model is deliberately not serializable (it never crosses a JSON
    boundary in these tests; ``model_dump`` would raise).
    """
    provider = ai.get_provider(
        "vercel",
        base_url=_BASE_URL,
        api_key=api_key,
        client=httpx.AsyncClient(transport=handler),
    )
    return ai.Model(model_id, provider_factory=lambda: provider)


mock_client = mock_model


def user_msg(text: str) -> messages.Message:
    return messages.Message(
        role="user",
        parts=[messages.TextPart(text=text)],
    )
