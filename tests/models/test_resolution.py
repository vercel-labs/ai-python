import pytest

import ai
from ai import ConfigurationError, models


def test_get_resolves_provider_qualified_model_id() -> None:
    model = ai.get_model("openai:gpt-5")

    assert model.id == "gpt-5"
    assert model.adapter == "openai"
    assert model.provider.name == "openai"


def test_get_resolves_provider_qualified_anthropic_model_id() -> None:
    model = models.get_model("anthropic:claude-sonnet-4-5")

    assert model.id == "claude-sonnet-4-5"
    assert model.adapter == "anthropic"
    assert model.provider.name == "anthropic"


def test_get_defaults_to_gateway_when_provider_is_omitted() -> None:
    model = models.get_model("anthropic/claude-sonnet-4")

    assert model.id == "anthropic/claude-sonnet-4"
    assert model.adapter == "ai-gateway-v3"
    assert model.provider.name == "ai-gateway"


def test_get_uses_default_model_env_when_model_id_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_SDK_DEFAULT_MODEL", "anthropic/claude-sonnet-4")

    model = models.get_model()

    assert model.id == "anthropic/claude-sonnet-4"
    assert model.adapter == "ai-gateway-v3"
    assert model.provider.name == "ai-gateway"


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
    assert provider.adapter == "openai"
    assert provider.default_base_url == "https://api.deepseek.com"
    assert provider.api_key_env == "DEEPSEEK_API_KEY"
    assert provider.config_envs == ()


def test_provider_base_class_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError, match="base class"):
        ai.Provider(name="custom", adapter="mock", base_url="https://example.com")


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
        provider.default_base_url == "https://${DATABRICKS_HOST}/ai-gateway/mlflow/v1"
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
    assert model.adapter == "ai-gateway-v3"
    assert model.provider.name == "ai-gateway"

    gateway_model = models.get_model("gateway:alibaba/qwen-3-14b")
    assert gateway_model.id == model.id
    assert gateway_model.adapter == model.adapter
    assert gateway_model.provider.name == model.provider.name


def test_get_uses_model_provider_config_for_anthropic_compatibility() -> None:
    model = models.get_model("azure:claude-sonnet-4-5")

    assert model.id == "claude-sonnet-4-5"
    assert model.adapter == "anthropic"
    assert model.provider.name == "azure"
    assert model.provider.default_base_url == (
        "https://${AZURE_RESOURCE_NAME}.services.ai.azure.com/anthropic/v1"
    )
    assert model.provider.api_key_env == "AZURE_API_KEY"
    assert model.provider.config_envs == ("AZURE_RESOURCE_NAME",)


def test_get_uses_model_provider_config_for_openai_compatibility() -> None:
    model = models.get_model("azure:kimi-k2.5")

    assert model.id == "kimi-k2.5"
    assert model.adapter == "openai"
    assert model.provider.name == "azure"
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
