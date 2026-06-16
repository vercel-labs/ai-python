"""Shared helpers for AI Gateway tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pydantic

import ai
from ai.types import messages

_BASE_URL = "https://gw.test/v3/ai"


class _GatewayProviderRef(ai.ProviderRef):
    _provider: ai.Provider[Any] = pydantic.PrivateAttr()

    def __init__(self, provider: ai.Provider[Any]) -> None:
        super().__init__("vercel")
        self._provider = provider

    def build(self) -> ai.Provider[Any]:
        return self._provider


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

    Per-test handlers are live objects, so this uses a test-only provider
    ref. It never crosses a JSON boundary in these tests.
    """
    provider = ai.get_provider(
        "vercel",
        base_url=_BASE_URL,
        api_key=api_key,
        client=httpx.AsyncClient(transport=handler),
    )
    return ai.Model(model_id, provider_ref=_GatewayProviderRef(provider))


mock_client = mock_model


def user_msg(text: str) -> messages.Message:
    return messages.Message(
        role="user",
        parts=[messages.TextPart(text=text)],
    )
