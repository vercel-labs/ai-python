"""Shared helpers for AI Gateway tests."""

from __future__ import annotations

import json
from typing import Any

import httpx

import ai
from ai.types import messages


def sse(*events: dict[str, Any]) -> str:
    """Build SSE response text from event dicts."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def mock_model(
    handler: httpx.MockTransport,
    *,
    model_id: str = "test-provider/test-model",
    api_key: str = "test-key",
) -> ai.Model:
    """Create a Gateway model wired to a mock transport."""
    provider = ai.get_provider(
        "vercel",
        base_url="https://gw.test/v3/ai",
        api_key=api_key,
        client=httpx.AsyncClient(transport=handler),
    )
    return ai.Model(model_id, provider=provider)


mock_client = mock_model


def user_msg(text: str) -> messages.Message:
    return messages.Message(
        role="user",
        parts=[messages.TextPart(text=text)],
    )
