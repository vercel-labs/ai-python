"""Model metadata types."""

import os
import weakref
from typing import TYPE_CHECKING, Any, Self, cast

import pydantic

if TYPE_CHECKING:
    from collections.abc import Callable

from ... import _modelsdev
from ...errors import ConfigurationError
from ...providers import base

_DEFAULT_MODEL_ENV = "AI_SDK_DEFAULT_MODEL"


def _exclude_if_none(v: Any) -> bool:
    # pydantic-core's SchemaSerializer doesn't GC-traverse ``exclude_if``
    # callables, so a closure stored there directly forms an uncollectable
    # cycle back through this module; wrap it in weakref.proxy at the call
    # site to keep the serializer's reference weak.
    return v is None


class Model(pydantic.BaseModel):
    """Lightweight reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``provider`` — :class:`Provider` that owns this model.
    * ``protocol`` — optional wire-protocol override for this model.
    """

    id: str
    provider: base.Provider[Any]
    protocol: base.ProviderProtocol[Any] | None = pydantic.Field(
        default=None,
        exclude_if=cast(
            "Callable[[Any], bool]", weakref.proxy(_exclude_if_none)
        ),
    )

    def __repr__(self) -> str:
        return f"Model(id={self.id!r}, provider={self.provider!r})"

    def __hash__(self) -> int:
        return hash((self.id, self.provider, self.protocol))

    def with_protocol(self, protocol: base.ProviderProtocol[Any]) -> Self:
        return self.model_copy(update={"protocol": protocol})


def get_model(
    model_id: str | None = None,
    *,
    protocol: base.ProviderProtocol[Any] | None = None,
) -> Model:
    """Resolve a model ID into a :class:`Model`.

    Args:
        model_id:
            Model ID, optionally in the format of ``"provider:model"``.
            When the provider is omitted, the model is routed through
            Vercel AI Gateway. Examples: ``"openai:gpt-5"`` or
            ``"anthropic/claude-sonnet-4"``. When omitted, reads
            ``AI_SDK_DEFAULT_MODEL`` from the environment.
        protocol:
            Optional wire-protocol override for this model. When omitted,
            the provider chooses its default protocol.

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

    model_info = _modelsdev.get_model_by_id(
        f"{provider_id}:{provider_model_id}"
    )
    model_provider_config = (
        None if model_info is None else model_info.provider_config
    )

    provider = base.Provider.from_id(
        provider_id,
        model_provider_config=model_provider_config,
    )

    return Model(id=provider_model_id, provider=provider, protocol=protocol)
