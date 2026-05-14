"""Anthropic ``probe`` tests.

The status-code handling is shared with OpenAI and exhaustively tested
in ``tests/providers/openai/test_probe.py``. This file only confirms the
provider-specific 200 path so we know URL routing is wired up.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import ai
from ai.providers.anthropic import AnthropicCompatibleProvider


def _client_with_mock(
    status_code: int = 200,
    json_body: Any = None,
    base_url: str = "https://anthropic.test",
) -> ai.Model:
    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(json_body or {}).encode()
        return httpx.Response(status_code, content=body)

    provider = ai.get_provider(
        "anthropic",
        base_url=base_url,
        api_key="sk-test-key",
        client=httpx.AsyncClient(
            base_url=base_url,
            transport=httpx.MockTransport(_handler),
        ),
    )
    return ai.Model("claude-opus-4-6", provider=provider)


async def test_200_succeeds() -> None:
    model = _client_with_mock(200, {"id": "claude-opus-4-6", "type": "model"})
    await model.provider.probe(model)


async def test_model_not_found_raises_model_not_found() -> None:
    model = _client_with_mock(404)
    with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
        await model.provider.probe(model)

    assert exc_info.value.model_id == model.id


async def test_custom_anthropic_version_header() -> None:
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        body = json.dumps({"id": "custom-model", "type": "model"}).encode()
        return httpx.Response(200, content=body)

    provider = AnthropicCompatibleProvider(
        name="custom-anthropic",
        default_base_url="https://anthropic.test",
        api_key="sk-test-key",
        anthropic_version="2024-01-01",
        headers={"X-Custom-Header": "example"},
        client=httpx.AsyncClient(
            base_url="https://anthropic.test",
            transport=httpx.MockTransport(_handler),
        ),
    )

    model = ai.Model("custom-model", provider=provider)
    await provider.probe(model)
    assert captured_headers["anthropic-version"] == "2024-01-01"
    assert captured_headers["x-custom-header"] == "example"
