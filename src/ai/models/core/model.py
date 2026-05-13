"""Model metadata types."""

import os

from ... import _modelsdev
from ...errors import ConfigurationError
from ...providers import base

_DEFAULT_MODEL_ENV = "AI_SDK_DEFAULT_MODEL"


class Model:
    """Lightweight reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``adapter`` — wire protocol key (e.g. ``"ai-gateway-v3"``, ``"anthropic"``).
    * ``provider`` — :class:`Provider` that owns this model.
    """

    def __init__(
        self,
        id: str,
        *,
        provider: base.Provider,
        adapter: str | None = None,
    ) -> None:
        self.id = id
        self.provider = provider
        self.adapter = adapter or provider.adapter

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Model)
            and self.id == other.id
            and self.adapter == other.adapter
            and self.provider is other.provider
        )

    def __repr__(self) -> str:
        return (
            f"Model(id={self.id!r}, adapter={self.adapter!r}, "
            f"provider={self.provider!r})"
        )

    def __hash__(self) -> int:
        return hash((self.id, self.adapter, id(self.provider)))

    async def check(self) -> bool:
        """Check whether this model is reachable and available.

        Returns ``True`` when credentials are valid and the model exists.
        Non-auth transport errors are raised so callers can distinguish
        misconfiguration from provider outages.
        """
        return await self.provider.check(self)


def get_model(model_id: str | None = None) -> Model:
    """Resolve a model ID into a :class:`Model`.

    Args:
        model_id:
            Model ID, optionally in the format of ``"provider:model"``.
            When the provider is omitted, the model is routed through
            Vercel AI Gateway. Examples: ``"openai:gpt-5"`` or
            ``"anthropic/claude-sonnet-4"``. When omitted, reads
            ``AI_SDK_DEFAULT_MODEL`` from the environment.
    Raises:
        Raises :class:`ai.ConfigurationError` when ``model_id`` and
        ``AI_SDK_DEFAULT_MODEL`` is empty or malformed.
        Raises a :class:`ai.UnsupportedProviderError` when the provider is
        unrecognized or otherwise unsupported.
    """
    if model_id is None:
        model_id = os.environ.get(_DEFAULT_MODEL_ENV)
        if not model_id:
            raise ConfigurationError(
                f"{_DEFAULT_MODEL_ENV} must be set when ai.get_model() "
                "is called without arguments"
            )

    if not model_id:
        raise ConfigurationError(f"get_model: malformed model_id: {model_id!r}")

    if ":" not in model_id:
        model_id = f"gateway:{model_id}"

    ref = _modelsdev.parse_model_id(model_id)
    assert ref.provider_id is not None  # guaranteed to be fully-qualified here
    provider_id = ref.provider_id
    provider_model_id = ref.model_id

    model_info = _modelsdev.get_model_by_id(f"{provider_id}:{provider_model_id}")
    model_provider_config = None if model_info is None else model_info.provider_config

    provider = base.Provider.from_id(
        provider_id,
        model_provider_config=model_provider_config,
    )

    return Model(provider_model_id, provider=provider)
