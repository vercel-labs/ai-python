from __future__ import annotations

import httpx
import pytest

import ai
from ai.providers.ai_gateway.client import errors


async def test_list_models_gets_config_with_gateway_headers_and_sorts_ids() -> (
    None
):
    captured_urls: list[str] = []
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "models": [
                    {"id": "openai/gpt-z"},
                    {"id": "anthropic/claude-a"},
                ]
            },
        )

    provider = ai.get_provider(
        "vercel",
        base_url="https://gateway.test/v3/ai",
        api_key="sk-test",
        headers={"X-Custom-Header": "example"},
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        ids = await provider.list_models()
    finally:
        await provider.aclose()

    assert captured_urls == ["https://gateway.test/v3/ai/config"]
    assert captured_headers["authorization"] == "Bearer sk-test"
    assert captured_headers["ai-gateway-protocol-version"] == "0.0.1"
    assert captured_headers["x-custom-header"] == "example"
    assert ids == ["anthropic/claude-a", "openai/gpt-z"]


async def test_list_models_remaps_gateway_errors() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {"message": "bad key", "type": "authentication_error"}
            },
        )

    provider = ai.get_provider(
        "vercel",
        base_url="https://gateway.test/v3/ai",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        with pytest.raises(ai.ProviderAuthenticationError) as exc_info:
            await provider.list_models()
    finally:
        await provider.aclose()

    assert isinstance(
        exc_info.value.__cause__, errors.GatewayAuthenticationError
    )
