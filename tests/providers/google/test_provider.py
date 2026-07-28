from __future__ import annotations

import importlib

import google.genai
import httpx
import pytest

import ai
from ai.providers.google import (
    GoogleGenerateContentProtocol,
    GoogleProvider,
)


async def test_list_models_strips_prefix_and_sorts_ids() -> None:
    captured_urls: list[str] = []
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "models/gemini-z"},
                    {"name": "models/gemini-a"},
                ]
            },
        )

    provider = ai.get_provider(
        "google",
        base_url="https://google.test",
        api_key="sk-test",
        headers={"X-Custom-Header": "example"},
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    ids = await provider.list_models()

    assert captured_urls[0].startswith("https://google.test/")
    assert "models" in captured_urls[0]
    assert captured_headers["x-goog-api-key"] == "sk-test"
    assert captured_headers["x-custom-header"] == "example"
    assert ids == ["gemini-a", "gemini-z"]


async def test_probe_maps_404_to_model_not_found() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"code": 404, "message": "not found"}},
        )

    provider = ai.get_provider(
        "google",
        base_url="https://google.test",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )
    model = ai.Model(id="gemini-nope", provider=provider)

    with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
        await provider.probe(model)

    assert exc_info.value.model_id == "gemini-nope"
    assert exc_info.value.provider == "google"


async def test_probe_unconfigured_raises_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = ai.get_provider("google")
    model = ai.Model(id="gemini-2.5-flash", provider=provider)

    with pytest.raises(ai.ProviderNotConfiguredError):
        await provider.probe(model)


async def test_get_provider_accepts_google_sdk_client() -> None:
    sdk_client = google.genai.Client(api_key="sk-test")
    provider = ai.get_provider("google", client=sdk_client)

    assert isinstance(provider, GoogleProvider)
    assert provider.sdk_client is sdk_client
    assert provider.is_configured() is True


def test_base_url_defaults_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    assert (
        ai.get_provider("google").base_url
        == "https://generativelanguage.googleapis.com"
    )


def test_base_url_reads_google_gemini_base_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://proxy.example.com")
    assert ai.get_provider("google").base_url == "https://proxy.example.com"


def test_api_key_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert ai.get_provider("google").is_configured() is False

    monkeypatch.setenv("GOOGLE_API_KEY", "sk-canonical")
    provider = ai.get_provider("google")
    assert provider.api_key == "sk-canonical"
    assert provider.is_configured() is True

    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    assert ai.get_provider("google").api_key == "sk-gemini"

    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "sk-google")
    assert ai.get_provider("google").api_key == "sk-google"


def test_get_provider_raises_installation_error_when_google_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def _missing_google(name: str, package: str | None = None) -> object:
        if name == "google.genai" or name.startswith("google.genai."):
            raise ModuleNotFoundError(name=name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _missing_google)

    provider = ai.get_provider("google", api_key="sk-test")

    with pytest.raises(ai.InstallationError) as exc_info:
        _ = provider.client

    assert "could not import `google`" in str(exc_info.value)
    assert "required to use the google provider" in str(exc_info.value)
    assert "ai[google]" in str(exc_info.value)


def test_get_provider_accepts_base_url_and_api_key() -> None:
    provider = ai.get_provider(
        "google",
        base_url="https://custom.example.com",
        api_key="sk-custom",
        headers={"X-Custom-Header": "example"},
    )

    model = ai.Model(id="custom-model", provider=provider)
    assert repr(provider) == "google"
    assert isinstance(provider.protocol, GoogleGenerateContentProtocol)
    assert provider.base_url == "https://custom.example.com"
    assert provider.api_key == "sk-custom"
    assert provider.headers == {"X-Custom-Header": "example"}
    assert provider.is_configured() is True
    assert model.id == "custom-model"
