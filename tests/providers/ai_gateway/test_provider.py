from __future__ import annotations

import httpx
import pytest
from vercel import oidc as vercel_oidc

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


async def test_list_models_uses_oidc_on_vercel_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setattr(
        vercel_oidc,
        "get_vercel_oidc_token",
        lambda: "oidc-test-token",
    )
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={"models": [{"id": "anthropic/claude-a"}]},
        )

    provider = ai.get_provider(
        "vercel",
        base_url="https://gateway.test/v3/ai",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        ids = await provider.list_models()
    finally:
        await provider.aclose()

    assert ids == ["anthropic/claude-a"]
    assert captured_headers["authorization"] == "Bearer oidc-test-token"
    assert captured_headers["ai-gateway-auth-method"] == "oidc"


async def test_list_models_uses_oidc_token_env_without_vercel_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "pulled-oidc-token")
    monkeypatch.setattr(
        vercel_oidc,
        "get_vercel_oidc_token",
        lambda: "pulled-oidc-token",
    )
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"models": []})

    provider = ai.get_provider(
        "vercel",
        base_url="https://gateway.test/v3/ai",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        await provider.list_models()
    finally:
        await provider.aclose()

    assert captured_headers["authorization"] == "Bearer pulled-oidc-token"
    assert captured_headers["ai-gateway-auth-method"] == "oidc"


async def test_api_key_env_takes_precedence_over_oidc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "env-test-key")
    monkeypatch.setenv("VERCEL", "1")

    def _fail_oidc() -> str:
        pytest.fail("OIDC should not be fetched when an API key is set")

    monkeypatch.setattr(
        vercel_oidc,
        "get_vercel_oidc_token",
        _fail_oidc,
    )
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"models": []})

    provider = ai.get_provider(
        "vercel",
        base_url="https://gateway.test/v3/ai",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        await provider.list_models()
    finally:
        await provider.aclose()

    assert captured_headers["authorization"] == "Bearer env-test-key"
    assert captured_headers["ai-gateway-auth-method"] == "api-key"


async def test_explicit_api_key_takes_precedence_over_oidc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "env-test-key")
    monkeypatch.setenv("VERCEL", "1")

    def _fail_oidc() -> str:
        pytest.fail("OIDC should not be fetched when an API key is set")

    monkeypatch.setattr(
        vercel_oidc,
        "get_vercel_oidc_token",
        _fail_oidc,
    )
    captured_headers: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"models": []})

    provider = ai.get_provider(
        "vercel",
        base_url="https://gateway.test/v3/ai",
        api_key="explicit-test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )

    try:
        await provider.list_models()
    finally:
        await provider.aclose()

    assert captured_headers["authorization"] == "Bearer explicit-test-key"
    assert captured_headers["ai-gateway-auth-method"] == "api-key"


async def test_is_configured_on_vercel_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("VERCEL", "1")

    provider = ai.get_provider("vercel")
    try:
        assert provider.is_configured() is True
    finally:
        await provider.aclose()
