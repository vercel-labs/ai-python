"""Model metadata types."""

import importlib
import os
from typing import Any, Self, cast

import pydantic

from ... import _modelsdev
from ...errors import ConfigurationError
from ...providers import base

_DEFAULT_MODEL_ENV = "AI_SDK_DEFAULT_MODEL"


type ProtocolName = str


class ProtocolRef(pydantic.BaseModel):
    """JSON-safe reference to a built-in provider protocol."""

    name: ProtocolName

    model_config = pydantic.ConfigDict(frozen=True)

    def __init__(self, name: ProtocolName | None = None, **data: Any) -> None:
        if name is not None:
            data["name"] = name
        super().__init__(**data)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value

    def build(self) -> base.ProviderProtocol[Any]:
        match self.name:
            case "openai.responses":
                protocol: Any = importlib.import_module(
                    "ai.providers.openai"
                ).OpenAIResponsesProtocol
                return cast("base.ProviderProtocol[Any]", protocol())
            case "openai.chat_completions":
                protocol = importlib.import_module(
                    "ai.providers.openai"
                ).OpenAIChatCompletionsProtocol
                return cast("base.ProviderProtocol[Any]", protocol())
            case "anthropic.messages":
                protocol = importlib.import_module(
                    "ai.providers.anthropic"
                ).AnthropicMessagesProtocol
                return cast("base.ProviderProtocol[Any]", protocol())
            case "gateway.v3":
                protocol = importlib.import_module(
                    "ai.providers.ai_gateway"
                ).GatewayV3Protocol
                return cast("base.ProviderProtocol[Any]", protocol())
            case _:
                raise ConfigurationError(
                    f"unknown protocol reference: {self.name!r}"
                )

    def __hash__(self) -> int:
        return hash(self.name)


class ProviderRef(pydantic.BaseModel):
    """JSON-safe provider settings used by :class:`Model`."""

    id: str
    model_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = pydantic.Field(default_factory=dict)
    env: dict[str, str] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(frozen=True)

    def __init__(self, id: str | None = None, **data: Any) -> None:
        if id is not None:
            data["id"] = id
        super().__init__(**data)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"id": value}
        return value

    def build(self) -> base.Provider[Any]:
        model_provider_config = None
        if self.model_id is not None:
            model_info = _modelsdev.get_model_by_id(
                f"{self.id}:{self.model_id}"
            )
            if model_info is not None:
                model_provider_config = model_info.provider_config
        return base.Provider.from_id(
            self.id,
            model_provider_config=model_provider_config,
            base_url=self.base_url,
            api_key=self.api_key,
            headers=self.headers,
            env=self.env,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.id,
                self.model_id,
                self.base_url,
                self.api_key,
                tuple(sorted(self.headers.items())),
                tuple(sorted(self.env.items())),
            )
        )


class Model(pydantic.BaseModel):
    """Reference to a model on a specific provider.

    * ``id`` — identifier sent to the provider (e.g. ``"claude-sonnet-4-6"``).
    * ``provider_ref`` — JSON-safe provider settings.
    * ``protocol_ref`` — optional JSON-safe wire-protocol override.

    The provider is built lazily on first :attr:`provider` access and
    cached.
    """

    id: str
    provider_ref: ProviderRef = pydantic.Field(
        validation_alias=pydantic.AliasChoices("provider_ref", "provider"),
        serialization_alias="provider",
    )
    protocol_ref: ProtocolRef | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("protocol_ref", "protocol"),
        serialization_alias="protocol",
    )

    _provider_instance: base.Provider[Any] | None = pydantic.PrivateAttr(
        default=None
    )
    _protocol_instance: base.ProviderProtocol[Any] | None = (
        pydantic.PrivateAttr(default=None)
    )

    def __init__(
        self,
        id: str,
        *,
        provider: ProviderRef | str | None = None,
        protocol: ProtocolRef | ProtocolName | None = None,
        provider_ref: ProviderRef | str | None = None,
        protocol_ref: ProtocolRef | ProtocolName | None = None,
    ) -> None:
        provider_ref = provider_ref if provider_ref is not None else provider
        protocol_ref = protocol_ref if protocol_ref is not None else protocol
        if provider_ref is None:
            raise TypeError("Model requires a provider")
        super().__init__(
            id=id,
            provider_ref=provider_ref,
            protocol_ref=protocol_ref,
        )

    model_config = pydantic.ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
    )

    @pydantic.field_validator("provider_ref", mode="before")
    @classmethod
    def _coerce_provider_ref(cls, value: Any) -> Any:
        if isinstance(value, str):
            return ProviderRef(value)
        return value

    @pydantic.field_validator("protocol_ref", mode="before")
    @classmethod
    def _coerce_protocol_ref(cls, value: Any) -> Any:
        if isinstance(value, str):
            return ProtocolRef(value)
        return value

    @pydantic.field_serializer("provider_ref")
    def _serialize_provider_ref(
        self,
        value: ProviderRef,
        info: pydantic.FieldSerializationInfo,
    ) -> dict[str, Any]:
        if type(value) is not ProviderRef:
            raise ConfigurationError(
                "custom provider refs cannot be serialized"
            )
        return value.model_dump(
            mode=info.mode,
            exclude_defaults=True,
            exclude_none=True,
        )

    @pydantic.field_serializer("protocol_ref")
    def _serialize_protocol_ref(
        self,
        value: ProtocolRef | None,
        info: pydantic.FieldSerializationInfo,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if type(value) is not ProtocolRef:
            raise ConfigurationError(
                "custom protocol refs cannot be serialized"
            )
        return value.model_dump(
            mode=info.mode,
            exclude_defaults=True,
            exclude_none=True,
        )

    @property
    def serializable(self) -> bool:
        """Whether this model round-trips through ``model_dump``."""
        return type(self.provider_ref) is ProviderRef and (
            self.protocol_ref is None or type(self.protocol_ref) is ProtocolRef
        )

    @property
    def provider(self) -> base.Provider[Any]:
        """Provider instance, built lazily from the provider ref and cached."""
        provider = self._provider_instance
        if provider is None:
            provider = self.provider_ref.build()
            if not isinstance(provider, base.Provider):
                raise ConfigurationError(
                    f"provider ref {self.provider_ref!r} returned "
                    f"{type(provider).__name__}, expected a Provider"
                )
            self._provider_instance = provider
        return provider

    @property
    def protocol(self) -> base.ProviderProtocol[Any] | None:
        """Protocol override instance, built lazily and cached."""
        if self.protocol_ref is None:
            return None
        protocol = self._protocol_instance
        if protocol is None:
            protocol = self.protocol_ref.build()
            if not isinstance(protocol, base.ProviderProtocol):
                raise ConfigurationError(
                    f"protocol ref {self.protocol_ref!r} returned "
                    f"{type(protocol).__name__}, expected a ProviderProtocol"
                )
            self._protocol_instance = protocol
        return protocol

    async def aclose(self) -> None:
        """Close the provider if this model lazily built one."""
        if self._provider_instance is not None:
            await self._provider_instance.aclose()
            self._provider_instance = None

    def __eq__(self, other: object) -> bool:
        # Pydantic's default __eq__ also compares private attributes.
        # Compare the JSON recipe only, so a lazily built provider does
        # not affect equality.
        return (
            isinstance(other, Model)
            and self.id == other.id
            and self.provider_ref == other.provider_ref
            and self.protocol_ref == other.protocol_ref
        )

    def __hash__(self) -> int:
        return hash((self.id, self.provider_ref, self.protocol_ref))

    def with_protocol(
        self,
        protocol: ProtocolRef | ProtocolName,
    ) -> Self:
        model = self.__class__(
            self.id,
            provider_ref=self.provider_ref,
            protocol_ref=protocol,
        )
        # Keep sharing an already-built provider instance.
        model._provider_instance = self._provider_instance
        return model


def get_model(
    model_id: str | None = None,
    *,
    protocol: ProtocolRef | ProtocolName | None = None,
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

    # Fail early on unknown or unsupported providers without building a
    # provider (and its client); the model only stores the recipe.
    base.Provider.resolve_type(
        provider_id, model_provider_config=model_provider_config
    )

    return Model(
        provider_model_id,
        provider=ProviderRef(provider_id, model_id=provider_model_id),
        protocol=protocol,
    )
