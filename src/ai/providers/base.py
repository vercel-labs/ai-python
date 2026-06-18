"""Base provider implementation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, cast

import pydantic
from typing_extensions import (
    TypeVar,
)

from .. import _modelsdev
from ..errors import UnsupportedProviderError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    import modelsdotdev

    from ..models.core import model as model_
    from ..models.core import params as params_
    from ..types import events
    from ..types import messages as messages_
    from ..types import tools as tools_

ClientT = TypeVar("ClientT", default=Any)


def _generic_origin(cls: type[Any]) -> type[Any]:
    # if this is a generic class, return the original class
    return (
        getattr(cls, "__pydantic_generic_metadata__", {}).get("origin") or cls
    )


def _class_id_default(
    cls: type[pydantic.BaseModel],
    field_name: str,
) -> str | None:
    if field_name not in getattr(cls, "__annotations__", {}):
        return None
    field = cls.model_fields.get(field_name)
    if field is None or not isinstance(field.default, str):
        return None
    return field.default


def _register_class_id[T](
    registry: dict[str, type[T]],
    class_id: str,
    cls: type[T],
    label: str,
) -> None:
    existing = registry.get(class_id)
    if existing is not None and existing is not cls:
        raise RuntimeError(f"duplicate {label} class id: {class_id!r}")
    registry[class_id] = cls


class ProviderProtocol(pydantic.BaseModel, Generic[ClientT]):
    """Interface implemented by provider wire protocols."""

    protocol_class_id: str  # used to restore the concrete protocol class

    model_config = pydantic.ConfigDict(frozen=True)

    def __init__(self, **data: Any) -> None:
        if _generic_origin(type(self)) is ProviderProtocol:
            raise TypeError("ProviderProtocol must be subclassed")
        super().__init__(**data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201
        super().__pydantic_init_subclass__(**kwargs)
        if _generic_origin(cls) is not cls:
            return
        protocol_class_id = _class_id_default(cls, "protocol_class_id")
        if protocol_class_id is not None:
            _register_class_id(
                _PROTOCOL_REGISTRY,
                protocol_class_id,
                cls,
                "provider protocol",
            )

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _load_registered_protocol(
        cls,
        value: Any,
        handler: pydantic.ModelWrapValidatorHandler[Self],
    ) -> Any:
        if isinstance(value, ProviderProtocol):
            return value
        if _generic_origin(cls) is not ProviderProtocol:
            return handler(value)
        if not isinstance(value, Mapping):
            return handler(value)

        protocol_class_id = value.get("protocol_class_id")
        if not isinstance(protocol_class_id, str):
            raise ValueError(
                "provider protocol data must include protocol_class_id"
            )
        protocol_type = _PROTOCOL_REGISTRY.get(protocol_class_id)
        if protocol_type is None:
            raise ValueError(
                f"unknown provider protocol_class_id: {protocol_class_id!r}"
            )
        return protocol_type.model_validate(value)

    def stream(
        self,
        client: ClientT,
        model: model_.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[events.Event]:
        """Stream a language-model response using *client*."""
        raise NotImplementedError(
            f"protocol {type(self).__name__!r} does not support stream()"
        )

    async def generate(
        self,
        client: ClientT,
        model: model_.Model,
        messages: list[messages_.Message],
        params: params_.GenerateParams,
        *,
        provider: str,
    ) -> messages_.Message:
        """Generate a non-streaming response using *client*."""
        raise NotImplementedError(
            f"protocol {type(self).__name__!r} does not support generate()"
        )


class Provider(pydantic.BaseModel, Generic[ClientT]):
    """Serializable provider configuration and base runtime interface.

    A provider carries provider-specific configuration: API endpoint,
    authentication, headers, environment overrides, and protocol selection.
    Model objects hold metadata plus a back-reference to their provider.

    Concrete provider subclasses add runtime behavior such as client creation,
    model listing, probing, generation, and streaming. Direct ``Provider(...)``
    construction is not allowed; define a subclass for custom providers.
    """

    handles: ClassVar[tuple[str, ...]] = ()

    provider_class_id: str  # used to restore the concrete provider class.
    name: str  # models.dev identity
    default_base_url: str
    protocol_override: pydantic.SerializeAsAny[ProviderProtocol[Any] | None] = (
        pydantic.Field(default=None, exclude_if=lambda v: v is None)
    )
    api_key_value: str | None = pydantic.Field(
        default=None, exclude_if=lambda v: v is None
    )
    api_key_env: str | None = None
    base_url_env: str | None = None
    config_envs: tuple[str, ...] = ()
    headers: dict[str, str] = pydantic.Field(default_factory=dict)
    env: dict[str, str] = pydantic.Field(default_factory=dict)

    _client: ClientT | None = pydantic.PrivateAttr(default=None)

    model_config = pydantic.ConfigDict(
        extra="allow",
        populate_by_name=True,
    )

    def __init__(self, **data: Any) -> None:
        if _generic_origin(type(self)) is Provider:
            raise TypeError("Provider must be subclassed")
        super().__init__(**data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:  # noqa: PLW3201
        super().__pydantic_init_subclass__(**kwargs)
        if _generic_origin(cls) is not cls:
            return

        provider_class_id = _class_id_default(cls, "provider_class_id")
        if provider_class_id is not None:
            _register_class_id(
                _PROVIDER_CLASS_REGISTRY,
                provider_class_id,
                cls,
                "provider",
            )

        for handle in cls.handles:
            existing = _PROVIDER_REGISTRY.get(handle)
            if existing is not None and existing is not cls:
                raise RuntimeError(f"duplicate provider handle: {handle!r}")
            _PROVIDER_REGISTRY[handle] = cls

    @pydantic.model_validator(mode="before")
    @classmethod
    def _normalize_config_data(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        if data.get("headers") is None:
            data["headers"] = {}
        if data.get("env") is None:
            data["env"] = {}
        if data.get("config_envs") is None:
            data["config_envs"] = ()
        return data

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _load_registered_provider(
        cls,
        value: Any,
        handler: pydantic.ModelWrapValidatorHandler[Self],
    ) -> Any:
        if isinstance(value, Provider):
            return value
        if _generic_origin(cls) is not Provider:
            return handler(value)
        if not isinstance(value, Mapping):
            return handler(value)

        provider_class_id = value.get("provider_class_id")
        if not isinstance(provider_class_id, str):
            raise ValueError("provider data must include provider_class_id")
        provider_type = _PROVIDER_CLASS_REGISTRY.get(provider_class_id)
        if provider_type is None:
            raise ValueError(
                f"unknown provider_class_id: {provider_class_id!r}"
            )
        return provider_type.model_validate(value)

    @property
    def base_url(self) -> str:
        """Default base URL for the provider API."""
        if self.base_url_env:
            base_url = (
                self.env.get(self.base_url_env)
                or os.environ.get(
                    self.base_url_env,
                )
                or self.default_base_url
            )
        else:
            base_url = self.default_base_url
        for env in self.config_envs:
            value = self.env.get(env) or os.environ.get(env)
            if value is not None:
                base_url = base_url.replace(f"${{{env}}}", value)
                base_url = base_url.replace(f"${env}", value)
        return base_url

    @property
    def api_key(self) -> str | None:
        """API key configured directly or via the provider's env var."""
        if self.api_key_value is not None:
            return self.api_key_value
        if self.api_key_env is None:
            return None
        return self.env.get(self.api_key_env) or os.environ.get(
            self.api_key_env
        )

    def is_configured(self) -> bool:
        """Return ``True`` when all required provider config is available."""
        if self.api_key_env is not None and not self.api_key:
            return False
        return all(self._config_value(env) for env in self.config_envs)

    def _config_value(self, env: str) -> str | None:
        return self.env.get(env) or os.environ.get(env)

    @property
    def client(self) -> ClientT:
        """Shared upstream client for this provider."""
        if self._client is None:
            raise RuntimeError("provider client has not been initialized")
        return self._client

    def _set_client(self, client: ClientT) -> None:
        self._client = client

    async def aclose(self) -> None:
        """Close provider-owned resources, if any."""
        return None

    @property
    def protocol(self) -> ProviderProtocol[ClientT]:
        """Default wire protocol used by this provider."""
        if self.protocol_override is not None:
            return cast("ProviderProtocol[ClientT]", self.protocol_override)
        return self.default_protocol()

    def default_protocol(self) -> ProviderProtocol[ClientT]:
        """Return this provider's default wire protocol."""
        raise RuntimeError(f"provider {self.name!r} does not have a protocol")

    async def list_models(self) -> list[str]:
        """List available model IDs from the provider API."""
        raise NotImplementedError

    def stream(
        self,
        model: model_.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events.Event]:
        """Stream a language-model response from this provider."""
        selected_protocol = model.protocol or self.protocol
        return selected_protocol.stream(
            self.client,
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
            provider=self.name,
        )

    async def generate(
        self,
        model: model_.Model,
        messages: list[messages_.Message],
        params: params_.GenerateParams,
    ) -> messages_.Message:
        """Generate a non-streaming response from this provider."""
        selected_protocol = model.protocol or self.protocol
        return await selected_protocol.generate(
            self.client,
            model,
            messages,
            params,
            provider=self.name,
        )

    async def probe(self, model: model_.Model) -> None:
        """Probe if provider is online and can serve given model.

        A probe function verifies that *model* can reach its provider and
        that it is available there. It returns successfully when credentials
        are valid **and** the model exists on the remote side.

        The check must be **free** — it should only hit metadata / listing
        endpoints that don't consume tokens or credits.

        Failures should raise provider errors; catch
        ``ProviderModelNotFoundError`` to distinguish missing models from
        other failures.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def from_id(
        cls,
        known_id: str,
        *,
        model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        client: Any | None = None,
        protocol: ProviderProtocol[Any] | None = None,
    ) -> Provider[Any]:
        """Return a concrete provider for a models.dev provider ID."""
        modelsdev_provider = _modelsdev.get_provider_by_id(known_id)
        if modelsdev_provider is None:
            raise ValueError(f"unknown provider id: {known_id!r}")

        for handle in (
            modelsdev_provider.id,
            _modelsdev.provider_npm(modelsdev_provider, model_provider_config),
        ):
            provider_type = _PROVIDER_REGISTRY.get(handle)
            if provider_type is not None:
                return provider_type.from_modelsdev_provider(
                    modelsdev_provider,
                    model_provider_config=model_provider_config,
                    base_url=base_url,
                    api_key=api_key,
                    headers=headers,
                    env=env,
                    client=client,
                    protocol=protocol,
                )

        raise UnsupportedProviderError(modelsdev_provider.id)

    @classmethod
    def from_modelsdev_provider(
        cls,
        provider: modelsdotdev.Provider,
        *,
        model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        client: Any | None = None,
        protocol: ProviderProtocol[Any] | None = None,
    ) -> Provider[Any]:
        """Construct this provider implementation from models.dev metadata."""
        raise NotImplementedError


_PROVIDER_REGISTRY: dict[str, type[Provider[Any]]] = {}
_PROVIDER_CLASS_REGISTRY: dict[str, type[Provider[Any]]] = {}
_PROTOCOL_REGISTRY: dict[str, type[ProviderProtocol[Any]]] = {}


def get_provider(
    id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    client: ClientT | None = None,
    protocol: ProviderProtocol[ClientT] | None = None,
) -> Provider[ClientT]:
    """Create a provider from a models.dev provider ID."""
    return Provider.from_id(
        id,
        base_url=base_url,
        api_key=api_key,
        headers=headers,
        env=env,
        client=client,
        protocol=protocol,
    )


def provider_config(
    provider: modelsdotdev.Provider,
    model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Return API key and config envs from models.dev data."""
    return _modelsdev.provider_config(provider, model_provider_config)


def provider_base_url(
    provider: modelsdotdev.Provider,
    model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
) -> str | None:
    """Return model-specific API URL override or provider API URL."""
    return _modelsdev.provider_base_url(provider, model_provider_config)
