from __future__ import annotations

import httpx
import pytest

from ai.models.core import client as client_
from ai.models.openai import openai


async def test_list_gets_models_with_auth_header_and_sorts_ids() -> None:
    captured_urls: list[str] = []
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-z"},
                    {"id": "gpt-a"},
                ]
            },
        )

    client = client_.Client(base_url="https://openai.test/v1", api_key="sk-test")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    try:
        ids = await openai.list(client=client)
    finally:
        await client.aclose()

    assert captured_urls == ["https://openai.test/v1/models"]
    assert captured_headers["authorization"] == "Bearer sk-test"
    assert ids == ["gpt-a", "gpt-z"]


def test_base_url_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert openai.base_url == "https://api.openai.com/v1"


def test_base_url_reads_openai_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    assert openai.base_url == "https://proxy.example.com/v1"


def test_client_uses_openai_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    c = openai.client()
    assert c.base_url == "https://proxy.example.com/v1"
    assert c.api_key == "sk-test"
