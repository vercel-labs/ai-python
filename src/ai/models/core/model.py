"""Model metadata types."""

import dataclasses
import os

from ... import _modelsdev
from ...errors import ConfigurationError
from ...providers import base
from .client import Client

_DEFAULT_MODEL_ENV = "AI_SDK_DEFAULT_MODEL"


@dataclasses.dataclass(frozen=True)
class Model:
    """Lightweight reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``adapter`` — wire protocol key (e.g. ``"ai-gateway-v3"``, ``"anthropic"``).
    * ``provider`` — :class:`Provider` that owns this model.
    * ``client`` — explicit :class:`Client` override (skips provider's default).
    """

    id: str
    adapter: str
    provider: base.Provider
    client: Client | None = dataclasses.field(default=None, repr=False)


def get_model(model_id: str | None = None, *, client: Client | None = None) -> Model:
    """Resolve a model ID into a :class:`Model`.

    Args:
        model_id:
            Model ID, optionally in the format of ``"provider:model"``.
            When the provider is omitted, the model is routed through the
            gateway. Examples: ``"openai:gpt-5"`` or
            ``"anthropic/claude-sonnet-4"``. When omitted, reads
            ``AI_SDK_DEFAULT_MODEL`` from the environment.
        client:
            Explicit client override. When omitted, the provider creates one
            from its default base URL and environment variables.

    Raises:
        Raises :class:`ai.ConfigurationError` when ``model_id`` is omitted and
        ``AI_SDK_DEFAULT_MODEL`` is not set.
        Raises a :class:`ai.UnsupportedProviderError` when the provider is
        unrecognized or otherwise unsupported.
    """
    if model_id is None:
        model_id = os.environ.get(_DEFAULT_MODEL_ENV)
        if not model_id:
            raise ConfigurationError(
                f"{_DEFAULT_MODEL_ENV} must be set to call get_model() "
                "without arguments"
            )

    if not model_id:
        raise ValueError("model_id must not be empty")
    if ":" in model_id:
        ref = _modelsdev.parse_model_id(model_id)
        if ref.provider_id is None:
            raise ValueError("model_id must include a known provider id")
        provider_id = ref.provider_id
        provider_model_id = ref.model_id
    else:
        provider_id = "gateway"
        provider_model_id = model_id

    model_info = _modelsdev.get_model_by_id(f"{provider_id}:{provider_model_id}")
    model_provider_config = None if model_info is None else model_info.provider_config
    return base.Provider.from_id(
        provider_id,
        model_provider_config=model_provider_config,
    )(provider_model_id, client=client)
