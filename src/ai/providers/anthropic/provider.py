"""Anthropic-compatible providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import httpx
import pydantic

from ... import errors as ai_errors
from .. import base
from . import _sdk, errors
from . import protocol as protocol_module
from . import tools as tools_module

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
    from types import ModuleType

    import anthropic
    import modelsdotdev

    from ...models.core import model as model_
    from ...models.core import params as params_
    from ...types import events
    from ...types import messages as messages_
    from ...types import tools as tools_

    AnthropicClient = httpx.AsyncClient | anthropic.AsyncAnthropic
    AnthropicSDKClient = anthropic.AsyncAnthropic
else:
    AnthropicClient = Any
    AnthropicSDKClient = Any

_BASE_URL = "https://api.anthropic.com"
_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
_API_KEY_ENV = "ANTHROPIC_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicCompatibleProvider(base.Provider[AnthropicSDKClient]):
    """Callable provider for Anthropic-compatible APIs."""

    handles: ClassVar[tuple[str, ...]] = ("anthropic", "@ai-sdk/anthropic")

    kind: Literal["anthropic-compatible"] = "anthropic-compatible"
    anthropic_version: str = _ANTHROPIC_VERSION

    _http_client: httpx.AsyncClient | None = pydantic.PrivateAttr(default=None)
    _close_client_on_aclose: bool = pydantic.PrivateAttr(default=False)
    _has_user_sdk_client: bool = pydantic.PrivateAttr(default=False)

    def __init__(
        self,
        *,
        name: str,
        default_base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        config_envs: Iterable[str] | None = None,
        anthropic_version: str = _ANTHROPIC_VERSION,
        headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        client: AnthropicClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
        **data: Any,
    ) -> None:
        if data:
            restore_data: dict[str, Any] = {
                **data,
                "name": name,
                "default_base_url": default_base_url,
                "api_key_env": api_key_env,
                "base_url_env": base_url_env,
                "config_envs": config_envs,
                "anthropic_version": anthropic_version,
                "headers": headers,
                "env": env,
            }
            if api_key is not None:
                restore_data["api_key"] = api_key
            if protocol is not None:
                restore_data["protocol"] = protocol
            super().__init__(**restore_data)
            self._http_client = None
            self._close_client_on_aclose = True
            self._has_user_sdk_client = False
            return

        anthropic_sdk = None
        if client is not None and not isinstance(client, httpx.AsyncClient):
            anthropic_sdk = _sdk.import_sdk(provider=name)

        if anthropic_sdk is not None and isinstance(
            client, anthropic_sdk.AsyncAnthropic
        ):
            sdk_client = client
            http_client = None
            has_user_sdk_client = True
        elif isinstance(client, httpx.AsyncClient) or client is None:
            sdk_client = None
            http_client = client
            has_user_sdk_client = False
        else:
            raise TypeError(
                "Anthropic providers require an httpx.AsyncClient or "
                "anthropic.AsyncAnthropic"
            )

        super().__init__(
            name=name,
            default_base_url=default_base_url,
            protocol=protocol,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            config_envs=config_envs,
            anthropic_version=anthropic_version,
            headers=headers,
            env=env,
        )
        self._has_user_sdk_client = has_user_sdk_client
        self._close_client_on_aclose = (
            sdk_client is None and http_client is None
        )
        self._http_client = http_client
        if sdk_client is not None:
            self._set_client(sdk_client)

    def _make_sdk_client(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> AnthropicSDKClient:
        anthropic_sdk = _sdk.import_sdk(provider=self.name)
        return anthropic_sdk.AsyncAnthropic(
            base_url=self.base_url,
            api_key=self.api_key or "",
            http_client=http_client,
            default_headers={
                **self.headers,
                "anthropic-version": self.anthropic_version,
            },
        )

    @property
    def sdk_client(self) -> AnthropicSDKClient:
        """Provider SDK client used for Anthropic-compatible API requests."""
        return self.client

    @property
    def client(self) -> AnthropicSDKClient:
        """Lazily-created SDK client for Anthropic-compatible requests."""
        if self._client is None:
            self._set_client(
                self._make_sdk_client(http_client=self._http_client)
            )
        return super().client

    def default_protocol(self) -> base.ProviderProtocol[AnthropicSDKClient]:
        """Return the default Anthropic-compatible protocol."""
        return protocol_module.AnthropicMessagesProtocol()

    def is_configured(self) -> bool:
        if self._has_user_sdk_client:
            return True
        if not self.api_key:
            return False
        return super().is_configured()

    async def aclose(self) -> None:
        """Close the provider-owned SDK client, if any."""
        if self._close_client_on_aclose and self._client is not None:
            await self.client.close()

    def stream(
        self,
        model: model_.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events.Event]:
        """Stream via the Anthropic messages protocol."""
        return super().stream(
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
        )

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
        client: AnthropicClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> base.Provider[AnthropicSDKClient]:
        resolved_base_url = base_url or base.provider_base_url(
            provider,
            model_provider_config,
        )
        if resolved_base_url is None and provider.id == "anthropic":
            resolved_base_url = _BASE_URL
        if resolved_base_url is None:
            raise ValueError(
                f"provider {provider.id!r} does not declare an API URL"
            )
        api_key_env, config_envs = base.provider_config(
            provider, model_provider_config
        )
        return cls(
            name=provider.id,
            default_base_url=resolved_base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url_env=_BASE_URL_ENV
            if provider.id == "anthropic" and base_url is None
            else None,
            config_envs=config_envs,
            headers=headers,
            env=env,
            client=client,
            protocol=protocol,
        )

    @property
    def tools(self) -> ModuleType:
        """The provider's built-in tool factories.

        Convenience accessor: ``anthropic.tools.web_search(...)``.
        """
        return tools_module

    async def list_models(self) -> list[str]:
        """List available model IDs from the Anthropic API."""
        anthropic_sdk = _sdk.import_sdk(provider=self.name)
        try:
            sdk_models = await self.sdk_client.models.list()
        except anthropic_sdk.AnthropicError as exc:
            raise errors.map_error(exc, provider=self.name) from exc
        return sorted(str(m.id) for m in sdk_models.data)

    async def probe(self, model: model_.Model) -> None:
        """Raise unless credentials are valid and the model exists."""
        if not self.is_configured():
            raise ai_errors.ProviderNotConfiguredError(
                f"provider {self.name!r} is not configured",
                provider=self.name,
            )
        anthropic_sdk = _sdk.import_sdk(provider=self.name)
        try:
            await self.sdk_client.models.retrieve(model.id)
        except anthropic_sdk.AnthropicError as exc:
            raise errors.map_error(
                exc,
                provider=self.name,
                model_id=model.id,
            ) from exc


__all__ = ["AnthropicCompatibleProvider"]
