from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx2
import pytest
import typesafe_sdk

import ai
from ai.providers.typesafe import TypeSafeProvider, TypeSafeSystemOneProtocol


def _provider(
    handler: Any, *, api_key: str | None = "ts-test"
) -> TypeSafeProvider:
    return TypeSafeProvider(
        api_key_value=api_key,
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _models_handler(
    names: list[str],
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://api.typesafe.ai/v1/models"
        return httpx2.Response(
            200,
            json={
                "models": [
                    {"name": name, "description": "", "release_date": ""}
                    for name in names
                ]
            },
        )

    return handler


def test_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    provider = TypeSafeProvider()

    assert provider.name == "typesafe"
    assert provider.base_url == "https://api.typesafe.ai"
    assert provider.api_key_env == "TYPESAFE_API_KEY"
    assert isinstance(provider.protocol, TypeSafeSystemOneProtocol)


def test_provider_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://typesafe.test")
    provider = TypeSafeProvider()

    assert not provider.is_configured()
    assert provider.base_url == "https://typesafe.test"

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-env")
    assert provider.is_configured()
    assert provider.api_key == "ts-env"


def test_missing_api_key_raises_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    provider = TypeSafeProvider()

    with pytest.raises(ai.ProviderNotConfiguredError, match="TYPESAFE_API_KEY"):
        _ = provider.client


def test_provider_round_trips_through_serialization() -> None:
    model = ai.Model(id="jev-latest", provider=TypeSafeProvider())
    restored = ai.Model.model_validate(model.model_dump(mode="json"))

    assert isinstance(restored.provider, TypeSafeProvider)
    assert restored == model


async def test_user_sdk_client_is_used_and_not_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    sdk_client = typesafe_sdk.AsyncTypeSafeClient(
        api_key="ts-user",
        transport=httpx2.MockTransport(_models_handler(["jev-latest"])),
    )
    provider = TypeSafeProvider(client=sdk_client)

    assert provider.is_configured()
    assert provider.client is sdk_client
    assert await provider.list_models() == ["jev-latest"]
    await provider.aclose()
    assert provider.client is sdk_client


async def test_aclose_closes_provider_created_client() -> None:
    provider = TypeSafeProvider(api_key_value="ts-test")
    client = provider.client

    await provider.aclose()

    assert provider.client is not client


async def test_list_models_sends_auth_and_headers() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(request.headers)
        return _models_handler(["jev-preview", "jev-latest"])(request)

    provider = TypeSafeProvider(
        api_key_value="ts-test",
        headers={"X-Custom": "1"},
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )

    assert await provider.list_models() == ["jev-latest", "jev-preview"]
    assert seen["authorization"] == "Bearer ts-test"
    assert seen["x-custom"] == "1"


async def test_list_models_maps_errors() -> None:
    provider = _provider(lambda request: httpx2.Response(401, json={}))

    with pytest.raises(ai.ProviderAuthenticationError) as exc_info:
        await provider.list_models()

    assert exc_info.value.provider == "typesafe"
    assert exc_info.value.http_context is not None
    assert exc_info.value.http_context.status_code == 401


async def test_probe_accepts_listed_model() -> None:
    provider = _provider(_models_handler(["jev-latest"]))

    await provider.probe(ai.Model(id="jev-latest", provider=provider))


async def test_probe_rejects_unlisted_model() -> None:
    provider = _provider(_models_handler(["jev-latest"]))

    with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
        await provider.probe(ai.Model(id="jev-nope", provider=provider))

    assert exc_info.value.model_id == "jev-nope"


async def test_probe_requires_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    provider = TypeSafeProvider()

    with pytest.raises(ai.ProviderNotConfiguredError):
        await provider.probe(ai.Model(id="jev-latest", provider=provider))
