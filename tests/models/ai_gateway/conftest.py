"""Shared helpers for AI Gateway tests."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai.models.core import client as client_
from ai.types import messages


def sse(*events: dict[str, Any]) -> str:
    """Build SSE response text from event dicts."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


def mock_client(
    handler: httpx.MockTransport, *, api_key: str = "test-key"
) -> client_.Client:
    """Create a Client wired to a mock transport."""
    c = client_.Client(base_url="https://gw.test/v4/ai", api_key=api_key)
    c._http = httpx.AsyncClient(transport=handler)
    return c


def user_msg(text: str) -> messages.Message:
    return messages.Message(
        role="user",
        parts=[messages.TextPart(text=text)],
    )
