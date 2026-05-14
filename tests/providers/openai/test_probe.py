from __future__ import annotations

import json
from typing import Any

import httpx
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


async def test_200_succeeds() -> None:
    model = _client_with_mock(200, {"id": "gpt-5.4", "object": "model"})
    await model.provider.probe(model)


@pytest.mark.parametrize(
    ("status", "error_cls"),
    [
        (401, ai.ProviderAuthenticationError),
        (403, ai.ProviderPermissionDeniedError),
        (404, ai.ProviderModelNotFoundError),
    ],
)
async def test_client_error_raises(
    status: int,
    error_cls: type[ai.ProviderAPIError],
) -> None:
    model = _client_with_mock(status)
    with pytest.raises(error_cls):
        await model.provider.probe(model)


async def test_500_raises() -> None:
    model = _client_with_mock(500)
    with pytest.raises(ai.ProviderInternalServerError):
        await model.provider.probe(model)


async def test_no_api_key_raises_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = ai.get_provider("openai", base_url="https://openai.test/v1")
    model = ai.Model("gpt-5.4", provider=provider)
    with pytest.raises(ai.ProviderNotConfiguredError):
        await provider.probe(model)
