from __future__ import annotations

import httpx
import pytest

from ai.models.anthropic import anthropic
from ai.models.core import client as client_


async def test_list_gets_models_with_provider_headers_and_sorts_ids() -> None:
    captured_urls: list[str] = []
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "claude-z"},
                    {"id": "claude-a"},
                ]
            },
        )

    client = client_.Client(base_url="https://anthropic.test", api_key="sk-test")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(_handler))

    try:
        ids = await anthropic.list(client=client)
    finally:
        await client.aclose()

    assert captured_urls == ["https://anthropic.test/v1/models"]
    assert captured_headers["x-api-key"] == "sk-test"
    assert captured_headers["anthropic-version"] == "2023-06-01"
    assert ids == ["claude-a", "claude-z"]


def test_base_url_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert anthropic.base_url == "https://api.anthropic.com"


def test_base_url_reads_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    assert anthropic.base_url == "https://proxy.example.com"


def test_client_uses_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = anthropic.client()
    assert c.base_url == "https://proxy.example.com"
    assert c.api_key == "sk-test"
