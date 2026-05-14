"""Base provider implementation."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic

from typing_extensions import TypeVar  # noqa: UP035 - default= is needed on 3.12

from .. import _modelsdev
from ..errors import UnsupportedProviderError

if TYPE_CHECKING:
    import anthropic
    import httpx
    import modelsdotdev
    import openai

    from ..models.core import model as model_

    ProviderClient = httpx.AsyncClient | openai.AsyncOpenAI | anthropic.AsyncAnthropic
else:
    ProviderClient = Any

ClientT = TypeVar("ClientT", default=Any)


class Provider(Generic[ClientT]):
    """Base class for model providers.

    A provider carries provider-specific configuration and a shared upstream
    client: API endpoint, authentication, and model enumeration. Model objects
    hold metadata (``id``, ``adapter``) plus a back-reference to their provider.
    """

    handles: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for handle in cls.handles:
            existing = _PROVIDER_REGISTRY.get(handle)
            if existing is not None and existing is not cls:
                raise RuntimeError(f"duplicate provider handle: {handle!r}")
            _PROVIDER_REGISTRY[handle] = cls

    def __init__(
        self,
        *,
        name: str,
        adapter: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        config_envs: Iterable[str] | None = None,
        headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        client: ClientT | None = None,
    ) -> None:
        if type(self) is Provider:
            raise TypeError("Provider is a base class; implement a subclass instead")
        self._name = name
        self._adapter = adapter
        self._base_url = base_url
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._base_url_env = base_url_env
        self._config_envs = tuple(config_envs or ())
        self._headers = dict(headers or {})
        self._env = dict(env or {})
        self._client = client

    @property
    def api_key_env(self) -> str | None:
        """Env var name that holds the API key (e.g. ``"OPENAI_API_KEY"``)."""
        return self._api_key_env

    @property
    def base_url_env(self) -> str | None:
        """Env var name that can override the default base URL."""
        return self._base_url_env

    @property
    def default_base_url(self) -> str:
        """Base URL configured on the provider before env overrides."""
        return self._base_url

    @property
    def base_url(self) -> str:
        """Default base URL for the provider API."""
        if self._base_url_env:
            base_url = (
                self._env.get(self._base_url_env)
                or os.environ.get(
                    self._base_url_env,
                )
                or self._base_url
            )
        else:
            base_url = self._base_url
        for env in self._config_envs:
            value = self._env.get(env) or os.environ.get(env)
            if value is not None:
                base_url = base_url.replace(f"${{{env}}}", value)
                base_url = base_url.replace(f"${env}", value)
        return base_url

    @property
    def api_key(self) -> str | None:
        """API key configured directly or via the provider's env var."""
        if self._api_key is not None:
            return self._api_key
        if self.api_key_env is None:
            return None
        return self._env.get(self.api_key_env) or os.environ.get(self.api_key_env)

    def is_configured(self) -> bool:
        """Return ``True`` when all required provider config is available."""
        if self.api_key_env is not None and not self.api_key:
            return False
        return all(self._config_value(env) for env in self.config_envs)

    @property
    def headers(self) -> dict[str, str]:
        """Custom headers sent with provider API requests."""
        return dict(self._headers)

    def _config_value(self, env: str) -> str | None:
        return self._env.get(env) or os.environ.get(env)

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
    def adapter(self) -> str:
        """Wire-protocol key used to look up stream/generate adapters."""
        return self._adapter

    @property
    def config_envs(self) -> tuple[str, ...]:
        """Additional env vars used to configure the provider client."""
        return self._config_envs

    @property
    def name(self) -> str:
        """Human-readable provider name (for repr, error messages)."""
        return self._name

    async def list(self) -> list[str]:
        """List available model IDs from the provider API."""
        raise NotImplementedError

    async def probe(self, model: model_.Model) -> None:
        """Probe if provider is online and can serve given model.

        A probe function verifies that *model* can reach its provider and that it
        is available there. It returns successfully when credentials are valid
        **and** the model exists on the remote side.

        The check must be **free** — it should only hit metadata / listing
        endpoints that don't consume tokens or credits.

        Failures should raise provider errors; catch ``ProviderModelNotFoundError``
        to distinguish missing models from other failures.
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
    ) -> Provider[Any]:
        """Construct this provider implementation from models.dev metadata."""
        raise NotImplementedError


_PROVIDER_REGISTRY: dict[str, type[Provider[Any]]] = {}


def get_provider(
    id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
    client: ProviderClient | None = None,
) -> Provider[Any]:
    """Create a provider from a models.dev provider ID."""
    return Provider.from_id(
        id,
        base_url=base_url,
        api_key=api_key,
        headers=headers,
        env=env,
        client=client,
    )


def provider_config(
    provider: modelsdotdev.Provider,
    model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Return ``api_key_env`` and non-secret config envs from models.dev data."""
    return _modelsdev.provider_config(provider, model_provider_config)


def provider_base_url(
    provider: modelsdotdev.Provider,
    model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
) -> str | None:
    """Return model-specific API URL override or provider API URL."""
    return _modelsdev.provider_base_url(provider, model_provider_config)
