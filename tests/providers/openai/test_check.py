from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest

import ai


def _client_with_mock(
    status_code: int = 200,
    json_body: Any = None,
    base_url: str = "https://openai.test/v1",
) -> ai.Model:
    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(json_body or {}).encode()
        return httpx.Response(status_code, content=body)

    provider = ai.get_provider(
        "openai",
        base_url=base_url,
        api_key="sk-test-key",
        client=httpx.AsyncClient(
            base_url=base_url,
            transport=httpx.MockTransport(_handler),
        ),
    )
    return ai.Model("gpt-5.4", provider=provider)


async def test_200_returns_true() -> None:
    model = _client_with_mock(200, {"id": "gpt-5.4", "object": "model"})
    assert await model.provider.check(model) is True


@pytest.mark.parametrize("status", [401, 403, 404])
async def test_client_error_returns_false(status: int) -> None:
    model = _client_with_mock(status)
    assert await model.provider.check(model) is False


async def test_500_raises() -> None:
    model = _client_with_mock(500)
    with pytest.raises(openai.APIStatusError):
        await model.provider.check(model)


async def test_no_api_key_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ai.get_provider("openai", base_url="https://openai.test/v1")
    model = ai.Model("gpt-5.4", provider=provider)
    assert await provider.check(model) is False
