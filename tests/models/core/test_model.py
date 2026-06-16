"""Tests for ``Model`` serialization and lazy provider construction."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic_core import PydanticSerializationError

import ai
from ai import models
from ai.providers.openai import OpenAIChatCompletionsProtocol

from ...conftest import MockProvider


def make_mock_provider(name: str = "mock") -> MockProvider:
    """Module-level provider factory used by serialization tests."""
    return MockProvider(name=name)


def not_a_provider() -> str:
    return "nope"


def test_model_dumps_factory_as_import_reference() -> None:
    model = models.Model("mock-model", provider_factory=make_mock_provider)

    data = model.model_dump(mode="json")

    assert data == {
        "id": "mock-model",
        "provider_factory": ("tests.models.core.test_model:make_mock_provider"),
        "provider_args": {},
        "protocol_factory": None,
        "protocol_args": {},
    }
    # the dump is real JSON
    json.dumps(data)


def test_model_json_round_trip() -> None:
    model = models.Model(
        "mock-model",
        provider_factory=make_mock_provider,
        provider_args={"name": "custom"},
    )

    restored = models.Model.model_validate(
        json.loads(json.dumps(model.model_dump(mode="json")))
    )

    assert restored == model
    assert hash(restored) == hash(model)
    assert restored.provider_factory is make_mock_provider
    assert restored.provider.name == "custom"


def test_get_model_round_trip_builds_equivalent_provider() -> None:
    model = ai.get_model("openai:gpt-5")
    restored = models.Model.model_validate(model.model_dump(mode="json"))

    assert restored == model
    assert restored.provider.name == "openai"


def test_provider_is_built_lazily_and_cached() -> None:
    model = models.Model("mock-model", provider_factory=make_mock_provider)

    assert model._provider_instance is None
    provider = model.provider
    assert isinstance(provider, MockProvider)
    assert model.provider is provider


async def test_aclose_drops_cached_provider() -> None:
    model = models.Model("mock-model", provider_factory=make_mock_provider)
    provider = model.provider

    await model.aclose()

    assert model._provider_instance is None
    assert model.provider is not provider


def test_with_protocol_round_trips_and_shares_provider() -> None:
    model = models.Model("mock-model", provider_factory=make_mock_provider)
    provider = model.provider

    override = model.with_protocol(OpenAIChatCompletionsProtocol)

    assert override.provider is provider
    assert isinstance(override.protocol, OpenAIChatCompletionsProtocol)

    restored = models.Model.model_validate(override.model_dump(mode="json"))
    assert isinstance(restored.protocol, OpenAIChatCompletionsProtocol)


def test_accepts_string_factory_reference() -> None:
    model = models.Model(
        "mock-model",
        provider_factory="tests.models.core.test_model:make_mock_provider",
    )

    assert model.provider_factory is make_mock_provider


def test_closure_factory_works_in_process_but_does_not_dump() -> None:
    provider = MockProvider()
    model = models.Model("mock-model", provider_factory=lambda: provider)

    assert model.provider is provider
    assert model.serializable is False
    with pytest.raises(PydanticSerializationError, match="lambdas"):
        model.model_dump()
    with pytest.raises(PydanticSerializationError, match="lambdas"):
        model.model_dump(mode="json")


def test_factory_defined_inside_function_does_not_dump() -> None:
    def local_factory() -> MockProvider:
        return MockProvider()

    model = models.Model("mock-model", provider_factory=local_factory)

    assert isinstance(model.provider, MockProvider)
    assert model.serializable is False
    with pytest.raises(PydanticSerializationError, match="module-level"):
        model.model_dump()


def test_factory_defined_in_main_does_not_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(make_mock_provider, "__module__", "__main__")
    model = models.Model("mock-model", provider_factory=make_mock_provider)

    assert model.serializable is False
    with pytest.raises(PydanticSerializationError, match="__main__"):
        model.model_dump()


def test_bound_method_factory_does_not_dump() -> None:
    provider = MockProvider()
    model = models.Model("mock-model", provider_factory=provider.list_models)

    assert model.serializable is False
    with pytest.raises(PydanticSerializationError, match="same object"):
        model.model_dump()


def test_non_json_provider_args_do_not_dump() -> None:
    model = models.Model(
        "mock-model",
        provider_factory=ai.get_provider,
        provider_args={"id": "openai", "client": httpx.AsyncClient()},
    )

    assert model.serializable is False
    with pytest.raises(PydanticSerializationError, match="round-trip"):
        model.model_dump()


def test_serializable_is_true_for_factory_models() -> None:
    assert ai.get_model("openai:gpt-5").serializable is True
    assert (
        models.Model(
            "mock-model", provider_factory=make_mock_provider
        ).serializable
        is True
    )


def test_rejects_args_not_matching_factory_signature() -> None:
    with pytest.raises(ai.ConfigurationError, match="signature"):
        models.Model(
            "mock-model",
            provider_factory=make_mock_provider,
            provider_args={"unknown_arg": 1},
        )


def test_rejects_protocol_args_without_protocol_factory() -> None:
    with pytest.raises(ai.ConfigurationError, match="protocol_factory"):
        models.Model(
            "mock-model",
            provider_factory=make_mock_provider,
            protocol_args={"x": 1},
        )


def test_rejects_malformed_string_reference() -> None:
    with pytest.raises(ai.ConfigurationError, match="malformed"):
        models.Model("mock-model", provider_factory="no-colon-here")


def test_rejects_unimportable_string_reference() -> None:
    with pytest.raises(ai.ConfigurationError, match="cannot import"):
        models.Model(
            "mock-model", provider_factory="tests.no_such_module:factory"
        )


def test_factory_returning_non_provider_fails_on_access() -> None:
    model = models.Model("mock-model", provider_factory=not_a_provider)

    with pytest.raises(ai.ConfigurationError, match="expected a Provider"):
        _ = model.provider


def test_equality_ignores_cached_provider_instance() -> None:
    a = models.Model("mock-model", provider_factory=make_mock_provider)
    b = models.Model("mock-model", provider_factory=make_mock_provider)
    _ = a.provider  # cache an instance on one side only

    assert a == b
    assert hash(a) == hash(b)

    c = models.Model(
        "mock-model",
        provider_factory=make_mock_provider,
        provider_args={"name": "other"},
    )
    assert a != c
