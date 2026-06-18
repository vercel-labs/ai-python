from typing import Any, Literal

import pydantic
import pytest

import ai
from ai import ConfigurationError, models
from ai.providers.ai_gateway import GatewayV3Protocol
from ai.providers.anthropic import AnthropicMessagesProtocol
from ai.providers.openai import (
    OpenAIChatCompletionsProtocol,
    OpenAIResponsesProtocol,
)


def test_get_resolves_provider_qualified_model_id() -> None:
    model = ai.get_model("openai:gpt-5")

    assert model.id == "gpt-5"
    assert model.provider.name == "openai"
    assert isinstance(model.provider.protocol, OpenAIResponsesProtocol)


def test_get_resolves_provider_qualified_anthropic_model_id() -> None:
    model = models.get_model("anthropic:claude-sonnet-4-5")

    assert model.id == "claude-sonnet-4-5"
    assert model.provider.name == "anthropic"
    assert isinstance(model.provider.protocol, AnthropicMessagesProtocol)


def test_get_defaults_to_gateway_when_provider_is_omitted() -> None:
    model = models.get_model("anthropic/claude-sonnet-4")

    assert model.id == "anthropic/claude-sonnet-4"
    assert model.provider.name == "ai-gateway"
    assert isinstance(model.provider.protocol, GatewayV3Protocol)


def test_get_uses_default_model_env_when_model_id_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_SDK_DEFAULT_MODEL", "anthropic/claude-sonnet-4")

    model = models.get_model()

    assert model.id == "anthropic/claude-sonnet-4"
    assert model.provider.name == "ai-gateway"
    assert isinstance(model.provider.protocol, GatewayV3Protocol)


def test_get_rejects_missing_default_model_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_SDK_DEFAULT_MODEL", raising=False)

    with pytest.raises(ai.ConfigurationError, match="AI_SDK_DEFAULT_MODEL"):
        models.get_model()


def test_get_rejects_empty_default_model_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_SDK_DEFAULT_MODEL", "")

    with pytest.raises(ai.ConfigurationError, match="AI_SDK_DEFAULT_MODEL"):
        models.get_model()


def test_provider_from_id_resolves_openai_compatible_provider() -> None:
    provider = ai.get_provider("deepseek")

    assert provider.name == "deepseek"
    assert isinstance(provider.protocol, OpenAIChatCompletionsProtocol)
    assert provider.default_base_url == "https://api.deepseek.com"
    assert provider.api_key_env == "DEEPSEEK_API_KEY"
    assert provider.config_envs == ()


def test_provider_base_class_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="must be subclassed"):
        ai.Provider(name="custom", default_base_url="https://example.com")


def test_provider_protocol_base_class_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="must be subclassed"):
        ai.ProviderProtocol()


def test_provider_from_id_uses_template_envs_for_base_url() -> None:
    provider = ai.get_provider("cloudflare-workers-ai")

    assert provider.default_base_url == (
        "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1"
    )
    assert provider.api_key_env == "CLOUDFLARE_API_KEY"
    assert provider.config_envs == ("CLOUDFLARE_ACCOUNT_ID",)


def test_provider_from_id_detects_token_env_after_url_env() -> None:
    provider = ai.get_provider("databricks")

    assert (
        provider.default_base_url
        == "https://${DATABRICKS_HOST}/ai-gateway/mlflow/v1"
    )
    assert provider.api_key_env == "DATABRICKS_TOKEN"
    assert provider.config_envs == ("DATABRICKS_HOST",)


def test_provider_from_id_resolves_gateway_provider() -> None:
    assert ai.get_provider("vercel").name == "ai-gateway"


def test_provider_from_id_resolves_gateway_alias() -> None:
    assert ai.get_provider("ai-gateway").name == "ai-gateway"
    assert ai.get_provider("gateway").name == "ai-gateway"


def test_get_resolves_gateway_alias() -> None:
    model = models.get_model("ai-gateway:alibaba/qwen-3-14b")

    assert model.id == "alibaba/qwen-3-14b"
    assert model.provider.name == "ai-gateway"
    assert isinstance(model.provider.protocol, GatewayV3Protocol)

    gateway_model = models.get_model("gateway:alibaba/qwen-3-14b")
    assert gateway_model.id == model.id
    assert gateway_model.provider.name == model.provider.name
    assert isinstance(gateway_model.provider.protocol, GatewayV3Protocol)


def test_get_uses_model_provider_config_for_anthropic_compatibility() -> None:
    model = models.get_model("azure:claude-sonnet-4-5")

    assert model.id == "claude-sonnet-4-5"
    assert model.provider.name == "azure"
    assert isinstance(model.provider.protocol, AnthropicMessagesProtocol)
    assert model.provider.default_base_url == (
        "https://${AZURE_RESOURCE_NAME}.services.ai.azure.com/anthropic/v1"
    )
    assert model.provider.api_key_env == "AZURE_API_KEY"
    assert model.provider.config_envs == ("AZURE_RESOURCE_NAME",)


def test_get_uses_model_provider_config_for_openai_compatibility() -> None:
    model = models.get_model("azure:kimi-k2.5")

    assert model.id == "kimi-k2.5"
    assert model.provider.name == "azure"
    assert isinstance(model.provider.protocol, OpenAIChatCompletionsProtocol)
    assert model.provider.default_base_url == (
        "https://${AZURE_RESOURCE_NAME}.services.ai.azure.com/models"
    )
    assert model.provider.api_key_env == "AZURE_API_KEY"
    assert model.provider.config_envs == ("AZURE_RESOURCE_NAME",)


def test_provider_from_id_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider id"):
        ai.get_provider("missing-provider")


def test_provider_from_id_rejects_unsupported_provider_package() -> None:
    with pytest.raises(ai.UnsupportedProviderError) as exc_info:
        ai.get_provider("google")

    assert exc_info.value.provider_id == "google"


def test_get_rejects_unsupported_provider_package() -> None:
    with pytest.raises(ai.errors.UnsupportedProviderError):
        models.get_model("google:gemini-2.5-pro")


def test_get_rejects_empty_model_id() -> None:
    with pytest.raises(ConfigurationError, match="malformed model_id: ''"):
        models.get_model("")


def test_get_model_accepts_model_protocol_override() -> None:
    protocol = OpenAIChatCompletionsProtocol()
    model = models.get_model("openai:gpt-5", protocol=protocol)

    assert model.protocol is protocol
    assert isinstance(model.provider.protocol, OpenAIResponsesProtocol)


def test_get_provider_accepts_provider_protocol_override() -> None:
    protocol = OpenAIChatCompletionsProtocol()
    provider = ai.get_provider("openai", protocol=protocol)

    assert provider.protocol is protocol


def test_model_json_roundtrip_restores_gateway_provider() -> None:
    model = models.get_model("gateway:anthropic/claude-sonnet-4.6")

    restored = ai.Model.model_validate_json(model.model_dump_json())

    assert restored.id == "anthropic/claude-sonnet-4.6"
    assert restored.provider.name == "ai-gateway"
    assert isinstance(restored.provider, ai.providers.GatewayProvider)
    assert isinstance(restored.provider.protocol, GatewayV3Protocol)


def test_model_json_roundtrip_preserves_explicit_provider_config() -> None:
    provider = ai.get_provider(
        "openai",
        base_url="https://custom.example.com/v1",
        api_key="sk-custom",
        headers={"X-Custom-Header": "example"},
    )
    model = ai.Model("custom-model", provider=provider)

    restored = ai.Model.model_validate_json(model.model_dump_json())

    assert isinstance(restored.provider, ai.providers.OpenAICompatibleProvider)
    assert restored.provider.name == "openai"
    assert restored.provider.default_base_url == "https://custom.example.com/v1"
    assert restored.provider.base_url == "https://custom.example.com/v1"
    assert restored.provider.api_key == "sk-custom"
    assert restored.provider.headers == {"X-Custom-Header": "example"}


def test_model_json_roundtrip_preserves_model_protocol_override() -> None:
    model = models.get_model(
        "openai:gpt-5",
        protocol=OpenAIChatCompletionsProtocol(),
    )

    restored = ai.Model.model_validate_json(model.model_dump_json())

    assert isinstance(restored.protocol, OpenAIChatCompletionsProtocol)
    assert isinstance(restored.provider.protocol, OpenAIResponsesProtocol)


def test_model_json_roundtrip_preserves_provider_protocol_override() -> None:
    provider = ai.get_provider(
        "openai",
        protocol=OpenAIChatCompletionsProtocol(),
    )
    model = ai.Model("gpt-5", provider=provider)

    restored = ai.Model.model_validate_json(model.model_dump_json())

    assert isinstance(restored.provider.protocol, OpenAIChatCompletionsProtocol)
    assert restored.protocol is None


def test_model_json_roundtrip_supports_registered_custom_provider() -> None:
    class CustomProtocol(models.ProviderProtocol[Any]):
        protocol_class_id: Literal["test-custom-protocol"] = (
            "test-custom-protocol"
        )
        mode: str

    class CustomProvider(models.Provider[Any]):
        provider_class_id: Literal["test-custom-provider"] = (
            "test-custom-provider"
        )
        region: str

        def default_protocol(self) -> models.ProviderProtocol[Any]:
            return CustomProtocol(mode="default")

    model = ai.Model(
        "custom-model",
        provider=CustomProvider(
            name="custom",
            default_base_url="https://custom.example.com",
            protocol_override=CustomProtocol(mode="provider"),
            region="test-region",
        ),
        protocol=CustomProtocol(mode="model"),
    )

    restored = ai.Model.model_validate_json(model.model_dump_json())

    assert isinstance(restored.provider, CustomProvider)
    assert restored.provider.region == "test-region"
    assert isinstance(restored.provider.protocol, CustomProtocol)
    assert restored.provider.protocol.mode == "provider"
    assert isinstance(restored.protocol, CustomProtocol)
    assert restored.protocol.mode == "model"


def test_model_json_roundtrip_rejects_provider_without_class_id() -> None:
    with pytest.raises(
        pydantic.ValidationError,
        match="provider data must include provider_class_id",
    ):
        ai.Model.model_validate(
            {
                "id": "custom-model",
                "provider": {
                    "name": "custom",
                    "default_base_url": "https://custom.example.com",
                },
            }
        )


def test_model_json_roundtrip_rejects_unknown_provider_class_id() -> None:
    with pytest.raises(
        pydantic.ValidationError,
        match="unknown provider_class_id",
    ):
        ai.Model.model_validate(
            {
                "id": "custom-model",
                "provider": {
                    "provider_class_id": "missing-provider",
                    "name": "custom",
                    "default_base_url": "https://custom.example.com",
                },
            }
        )


def test_model_json_roundtrip_rejects_unknown_protocol_class_id() -> None:
    data = ai.get_model("openai:gpt-5").model_dump()
    data["protocol"] = {"protocol_class_id": "missing-protocol"}

    with pytest.raises(
        pydantic.ValidationError,
        match="unknown provider protocol_class_id",
    ):
        ai.Model.model_validate(data)


def test_model_json_roundtrip_rejects_protocol_without_class_id() -> None:
    data = ai.get_model("openai:gpt-5").model_dump()
    data["protocol"] = {}

    with pytest.raises(
        pydantic.ValidationError,
        match="provider protocol data must include protocol_class_id",
    ):
        ai.Model.model_validate(data)


def test_model_json_roundtrip_rejects_unknown_provider_protocol_class_id() -> (
    None
):
    provider = ai.get_provider(
        "openai",
        protocol=OpenAIChatCompletionsProtocol(),
    )
    data = ai.Model("gpt-5", provider=provider).model_dump()
    assert isinstance(data["provider"], dict)
    data["provider"]["protocol_override"] = {
        "protocol_class_id": "missing-protocol"
    }

    with pytest.raises(
        pydantic.ValidationError,
        match="unknown provider protocol_class_id",
    ):
        ai.Model.model_validate(data)
