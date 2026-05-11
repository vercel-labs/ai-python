from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai.models.ai_gateway import ai_gateway, check
from ai.models.core import client as client_

_MODEL = ai_gateway("anthropic/claude-opus-4-6")


def _gateway_client(
    *,
    credits_status: int = 200,
    config_status: int = 200,
    config_body: dict[str, Any] | None = None,
    api_key: str | None = "sk-test-key",
) -> client_.Client:
    credits_body = json.dumps({"balance": "10.00", "totalUsed": "5.00"})
    config_bytes = json.dumps(config_body or {"models": []}).encode()

    def _handler(request: httpx.Request) -> httpx.Response:
        if "/v1/credits" in str(request.url):
            return httpx.Response(credits_status, content=credits_body.encode())
        return httpx.Response(config_status, content=config_bytes)

    client = client_.Client(base_url="https://gateway.test/v4/ai", api_key=api_key)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return client


async def test_auth_ok_model_present() -> None:
    config = {"models": [{"id": "anthropic/claude-opus-4-6"}]}
    assert await check.check(_gateway_client(config_body=config), _MODEL) is True


async def test_auth_ok_model_absent() -> None:
    config = {"models": [{"id": "openai/gpt-5.4"}]}
    assert await check.check(_gateway_client(config_body=config), _MODEL) is False


@pytest.mark.parametrize("status", [401, 403])
async def test_credits_auth_error_returns_false(status: int) -> None:
    assert await check.check(_gateway_client(credits_status=status), _MODEL) is False


async def test_credits_500_raises() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        await check.check(_gateway_client(credits_status=500), _MODEL)
