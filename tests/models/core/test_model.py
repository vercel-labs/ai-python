"""Tests for ``Model`` serialization and lazy provider construction."""

from __future__ import annotations

import json
from typing import Any

import pydantic
import pytest
from pydantic_core import PydanticSerializationError

import ai
from ai import models
from ai.providers.openai import OpenAIChatCompletionsProtocol

from ...conftest import MOCK_PROVIDER, MockProvider


class _FreshMockProviderRef(models.ProviderRef):
    def __init__(self) -> None:
        super().__init__("mock")

    def build(self) -> MockProvider:
        return MockProvider()


class _Box(pydantic.BaseModel):
    model: models.Model


def test_model_dumps_provider_ref_as_json_data() -> None:
    model = models.Model(
        "mock-model",
        provider=models.ProviderRef("mock", base_url="http://mock.test"),
    )

    data = model.model_dump(mode="json")

    assert data == {
        "id": "mock-model",
        "provider": {
            "id": "mock",
            "base_url": "http://mock.test",
        },
        "protocol": None,
    }
    json.dumps(data)


def test_model_json_round_trip() -> None:
    model = ai.get_model("openai:gpt-5")

    restored = models.Model.model_validate(
        json.loads(json.dumps(model.model_dump(mode="json")))
    )

    assert restored == model
    assert hash(restored) == hash(model)
    assert restored.provider_ref == models.ProviderRef(
        "openai", model_id="gpt-5"
    )


def test_model_inside_pydantic_model_round_trips() -> None:
    box = _Box(model=ai.get_model("openai:gpt-5"))

    restored = _Box.model_validate(
        json.loads(json.dumps(box.model_dump(mode="json")))
    )

    assert restored == box


def test_model_validate_does_not_resolve_provider() -> None:
    model = models.Model.model_validate(
        {"id": "test-model", "provider": {"id": "not-registered"}}
    )

    assert model.provider_ref.id == "not-registered"
    with pytest.raises(ValueError, match="unknown provider"):
        _ = model.provider


def test_get_model_round_trip_builds_equivalent_provider() -> None:
    model = ai.get_model("openai:gpt-5")
    restored = models.Model.model_validate(model.model_dump(mode="json"))

    assert restored == model
    assert restored.provider.name == "openai"


def test_provider_is_built_lazily_and_cached() -> None:
    model = models.Model("mock-model", provider="mock")

    assert model._provider_instance is None
    assert model.provider is MOCK_PROVIDER
    assert model.provider is MOCK_PROVIDER


async def test_aclose_drops_cached_provider() -> None:
    model = models.Model("mock-model", provider_ref=_FreshMockProviderRef())
    provider = model.provider

    await model.aclose()

    assert model._provider_instance is None
    assert model.provider is not provider


def test_with_protocol_round_trips_and_shares_provider() -> None:
    model = models.Model("mock-model", provider="mock")
    provider = model.provider

    override = model.with_protocol("openai.chat_completions")

    assert override.provider is provider
    assert isinstance(override.protocol, OpenAIChatCompletionsProtocol)

    restored = models.Model.model_validate(override.model_dump(mode="json"))
    assert isinstance(restored.protocol, OpenAIChatCompletionsProtocol)


def test_accepts_string_provider_ref() -> None:
    model = models.Model("mock-model", provider="mock")

    assert model.provider_ref == models.ProviderRef("mock")


def test_accepts_string_protocol_ref() -> None:
    model = models.Model(
        "mock-model",
        provider="mock",
        protocol="openai.chat_completions",
    )

    assert model.protocol_ref == models.ProtocolRef("openai.chat_completions")


def test_unknown_protocol_fails_on_access_not_validation() -> None:
    model = models.Model.model_validate(
        {
            "id": "mock-model",
            "provider": {"id": "mock"},
            "protocol": {"name": "not-real"},
        }
    )

    with pytest.raises(ai.ConfigurationError, match="unknown protocol"):
        _ = model.protocol


def test_custom_provider_ref_works_in_process_but_does_not_dump() -> None:
    model = models.Model("mock-model", provider_ref=_FreshMockProviderRef())

    assert isinstance(model.provider, MockProvider)
    assert model.serializable is False
    with pytest.raises(PydanticSerializationError, match="provider refs"):
        model.model_dump()


def test_serializable_is_true_for_plain_refs() -> None:
    assert ai.get_model("openai:gpt-5").serializable is True
    assert models.Model("mock-model", provider="mock").serializable is True


def test_provider_ref_rejects_bad_header_type() -> None:
    with pytest.raises(pydantic.ValidationError):
        models.Model.model_validate(
            {
                "id": "mock-model",
                "provider": {
                    "id": "mock",
                    "headers": {"x-test": object()},
                },
            }
        )


def test_factory_returning_non_provider_fails_on_access() -> None:
    class NotAProviderRef(models.ProviderRef):
        def __init__(self) -> None:
            super().__init__("mock")

        def build(self) -> Any:
            return "nope"

    model = models.Model("mock-model", provider_ref=NotAProviderRef())

    with pytest.raises(ai.ConfigurationError, match="expected a Provider"):
        _ = model.provider


def test_equality_ignores_cached_provider_instance() -> None:
    a = models.Model("mock-model", provider="mock")
    b = models.Model("mock-model", provider="mock")
    _ = a.provider

    assert a == b
    assert hash(a) == hash(b)

    c = models.Model(
        "mock-model",
        provider=models.ProviderRef("mock", base_url="http://other.test"),
    )
    assert a != c
