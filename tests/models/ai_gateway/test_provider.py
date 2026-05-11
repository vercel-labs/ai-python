from __future__ import annotations

import httpx

from ai.models.ai_gateway import ai_gateway
from ai.models.core import client as client_


async def test_list_gets_config_with_gateway_headers_and_sorts_ids() -> None:
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

    client = client_.Client(base_url="https://gateway.test/v4/ai", api_key="sk-test")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    try:
        ids = await ai_gateway.list(client=client)
    finally:
        await client.aclose()

    assert captured_urls == ["https://gateway.test/v4/ai/config"]
    assert captured_headers["authorization"] == "Bearer sk-test"
    assert captured_headers["ai-gateway-protocol-version"] == "0.0.1"
    assert ids == ["anthropic/claude-a", "openai/gpt-z"]
