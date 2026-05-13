from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import ai

_MODEL_ID = "anthropic/claude-opus-4-6"


def _gateway_client(
    *,
    credits_status: int = 200,
    config_status: int = 200,
    config_body: dict[str, Any] | None = None,
    api_key: str | None = "sk-test-key",
) -> ai.Model:
    credits_body = json.dumps({"balance": "10.00", "totalUsed": "5.00"})
    config_bytes = json.dumps(config_body or {"models": []}).encode()

    def _handler(request: httpx.Request) -> httpx.Response:
        if "/v1/credits" in str(request.url):
            return httpx.Response(credits_status, content=credits_body.encode())
        return httpx.Response(config_status, content=config_bytes)

    provider = ai.get_provider(
        "vercel",
        base_url="https://gateway.test/v3/ai",
        api_key=api_key,
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )
    return ai.Model(_MODEL_ID, provider=provider)


async def test_auth_ok_model_present() -> None:
    config = {"models": [{"id": "anthropic/claude-opus-4-6"}]}
    model = _gateway_client(config_body=config)
    assert await model.provider.check(model) is True


async def test_auth_ok_model_absent() -> None:
    config = {"models": [{"id": "openai/gpt-5.4"}]}
    model = _gateway_client(config_body=config)
    assert await model.provider.check(model) is False


@pytest.mark.parametrize("status", [401, 403])
async def test_credits_auth_error_returns_false(status: int) -> None:
    model = _gateway_client(credits_status=status)
    assert await model.provider.check(model) is False


async def test_missing_configuration_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)

    model = _gateway_client(api_key=None)
    assert await model.provider.check(model) is False


async def test_credits_500_raises() -> None:
    model = _gateway_client(credits_status=500)
    with pytest.raises(httpx.HTTPStatusError):
        await model.provider.check(model)
