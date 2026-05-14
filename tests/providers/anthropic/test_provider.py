from __future__ import annotations

import importlib

import anthropic
import httpx
import pytest

import ai
from ai.providers.anthropic import AnthropicCompatibleProvider


async def test_list_models_gets_models_with_provider_headers_and_sorts_ids() -> None:
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

    provider = ai.get_provider(
        "anthropic",
        base_url="https://anthropic.test",
        api_key="sk-test",
        headers={"X-Custom-Header": "example"},
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        ids = await provider.list_models()
    finally:
        await provider.aclose()

    assert captured_urls == ["https://anthropic.test/v1/models"]
    assert captured_headers["x-api-key"] == "sk-test"
    assert captured_headers["anthropic-version"] == "2023-06-01"
    assert captured_headers["x-custom-header"] == "example"
    assert ids == ["claude-a", "claude-z"]


async def test_list_models_maps_sdk_errors_to_provider_hierarchy() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529,
            json={"error": {"message": "overloaded", "type": "overloaded_error"}},
            headers={"request-id": "req-anthropic"},
        )

    provider = ai.get_provider(
        "anthropic",
        base_url="https://anthropic.test",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        with pytest.raises(ai.ProviderOverloadedError) as exc_info:
            await provider.list_models()
    finally:
        await provider.aclose()

    exc = exc_info.value
    assert isinstance(exc, ai.ProviderError)
    assert isinstance(exc.__cause__, anthropic.APIStatusError)
    assert exc.provider == "anthropic"
    assert exc.http_context is not None
    assert exc.http_context.status_code == 529
    assert exc.http_context.request is not None
    assert exc.http_context.response is not None
    assert exc.request_id == "req-anthropic"
    assert exc.type == "overloaded_error"


async def test_list_models_404_stays_generic_not_found() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "missing"}})

    provider = ai.get_provider(
        "anthropic",
        base_url="https://anthropic.test",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        with pytest.raises(ai.ProviderNotFoundError) as exc_info:
            await provider.list_models()
    finally:
        await provider.aclose()

    assert not isinstance(exc_info.value, ai.ProviderModelNotFoundError)


async def test_get_provider_accepts_anthropic_sdk_client() -> None:
    sdk_client = anthropic.AsyncAnthropic(api_key="sk-test")
    provider = ai.get_provider("anthropic", client=sdk_client)

    try:
        assert isinstance(provider, AnthropicCompatibleProvider)
        assert provider.sdk_client is sdk_client
        assert provider.is_configured() is True
    finally:
        await sdk_client.close()


def test_base_url_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert ai.get_provider("anthropic").base_url == "https://api.anthropic.com"


def test_base_url_reads_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    assert ai.get_provider("anthropic").base_url == "https://proxy.example.com"


def test_provider_uses_anthropic_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    provider = ai.get_provider("anthropic")
    assert provider.base_url == "https://proxy.example.com"
    assert provider.api_key == "sk-test"
    assert provider.is_configured() is True


def test_provider_is_configured_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert ai.get_provider("anthropic").is_configured() is False
    assert ai.get_provider("anthropic", api_key="sk-test").is_configured() is True


def test_get_provider_raises_installation_error_when_anthropic_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def _missing_anthropic(name: str, package: str | None = None) -> object:
        if name == "anthropic" or name.startswith("anthropic."):
            raise ModuleNotFoundError(name="anthropic")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _missing_anthropic)

    with pytest.raises(ai.InstallationError) as exc_info:
        ai.get_provider("anthropic", api_key="sk-test")

    assert "could not import `anthropic`" in str(exc_info.value)
    assert "required to use the anthropic provider" in str(exc_info.value)
    assert "ai[anthropic]" in str(exc_info.value)


def test_get_provider_accepts_base_url_and_api_key() -> None:
    provider = ai.get_provider(
        "anthropic",
        base_url="https://custom.example.com",
        api_key="sk-custom",
        headers={"X-Custom-Header": "example"},
    )

    model = ai.Model("custom-model", provider=provider)
    assert repr(provider) == "anthropic"
    assert provider.adapter == "anthropic"
    assert provider.base_url == "https://custom.example.com"
    assert provider.api_key == "sk-custom"
    assert provider.headers == {"X-Custom-Header": "example"}
    assert provider.is_configured() is True
    assert model.id == "custom-model"
    assert model.adapter == "anthropic"
    assert model.provider is provider


def test_get_provider_env_overrides_base_url_env() -> None:
    provider = ai.get_provider(
        "anthropic",
        env={"ANTHROPIC_BASE_URL": "https://proxy.example.com"},
    )

    assert provider.base_url == "https://proxy.example.com"
