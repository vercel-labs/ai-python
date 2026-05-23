"""OpenAI-compatible providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from ... import errors as ai_errors
from .. import base
from . import _sdk, errors
from . import protocol as protocol_module
from . import tools as tools_module

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
    from types import ModuleType

    import modelsdotdev
    import openai
    import pydantic

    from ...models.core import model as model_
    from ...types import events
    from ...types import messages as messages_
    from ...types import tools as tools_

    OpenAIClient = httpx.AsyncClient | openai.AsyncOpenAI
    OpenAISDKClient = openai.AsyncOpenAI
else:
    OpenAIClient = Any
    OpenAISDKClient = Any

_BASE_URL = "https://api.openai.com/v1"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAICompatibleProvider(base.Provider[OpenAISDKClient]):
    """Provider configuration for OpenAI-compatible APIs."""

    handles: ClassVar[tuple[str, ...]] = (
        "openai",
        "@ai-sdk/openai",
        "@ai-sdk/openai-compatible",
    )

    def __init__(
        self,
        *,
        name: str,
        default_base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        config_envs: Iterable[str] | None = None,
        headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        client: OpenAIClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> None:
        openai_sdk = None
        if client is not None and not isinstance(client, httpx.AsyncClient):
            openai_sdk = _sdk.import_sdk(provider=name)

        if openai_sdk is not None and isinstance(
            client, openai_sdk.AsyncOpenAI
        ):
            sdk_client = client
            http_client = None
            self._has_user_sdk_client = True
        elif isinstance(client, httpx.AsyncClient) or client is None:
            sdk_client = None
            http_client = client
            self._has_user_sdk_client = False
        else:
            raise TypeError(
                "OpenAI providers require an httpx.AsyncClient or "
                "openai.AsyncOpenAI"
            )

        super().__init__(
            name=name,
            base_url=default_base_url,
            protocol=protocol or protocol_module.default_protocol(name),
            api_key=api_key,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            config_envs=config_envs,
            headers=headers,
            env=env,
        )
        self._close_client_on_aclose = (
            sdk_client is None and http_client is None
        )
        self._pending_http_client = http_client
        if sdk_client is not None:
            self._set_client(sdk_client)

    def _make_sdk_client(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> OpenAISDKClient:
        openai_sdk = _sdk.import_sdk(provider=self.name)
        return openai_sdk.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "",
            default_headers=self.headers,
            http_client=http_client,
        )

    @property
    def client(self) -> OpenAISDKClient:
        # Constructing the OpenAI SDK client imports httpx internals that
        # trip the Temporal workflow sandbox. Defer until first use so
        # that constructing a Provider (e.g. via ai.get_model) is safe in
        # a workflow; the actual HTTP client only materializes in
        # activities.
        if self._client is None:
            self._set_client(
                self._make_sdk_client(http_client=self._pending_http_client)
            )
        assert self._client is not None
        return self._client

    @property
    def sdk_client(self) -> OpenAISDKClient:
        """Provider SDK client used for OpenAI-compatible API requests."""
        return self.client

    def is_configured(self) -> bool:
        if self._has_user_sdk_client:
            return True
        if not self.api_key:
            return False
        return super().is_configured()

    async def aclose(self) -> None:
        """Close the provider-owned SDK client, if any."""
        if self._close_client_on_aclose and self._client is not None:
            await self._client.close()

    def stream(
        self,
        model: model_.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
    ) -> AsyncGenerator[events.Event]:
        """Stream via this provider's configured OpenAI-compatible protocol."""
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
        client: OpenAIClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> base.Provider[OpenAISDKClient]:
        resolved_base_url = base_url or base.provider_base_url(
            provider,
            model_provider_config,
        )
        if resolved_base_url is None and provider.id == "openai":
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
            if provider.id == "openai" and base_url is None
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

        Convenience accessor: ``openai.tools.web_search(...)``. These tools
        require a protocol that supports OpenAI provider-executed tools, such
        as :class:`OpenAIResponsesProtocol`.
        """
        return tools_module

    async def list_models(self) -> list[str]:
        """List available model IDs from the OpenAI-compatible API."""
        openai_sdk = _sdk.import_sdk(provider=self.name)
        try:
            sdk_models = await self.sdk_client.models.list()
        except openai_sdk.OpenAIError as exc:
            raise errors.map_error(exc, provider=self.name) from exc
        return sorted(str(m.id) for m in sdk_models.data)

    async def probe(self, model: model_.Model) -> None:
        """Raise unless credentials are valid and the model exists."""
        if not self.is_configured():
            raise ai_errors.ProviderNotConfiguredError(
                f"provider {self.name!r} is not configured",
                provider=self.name,
            )
        openai_sdk = _sdk.import_sdk(provider=self.name)
        try:
            await self.sdk_client.models.retrieve(model.id)
        except openai_sdk.OpenAIError as exc:
            raise errors.map_error(
                exc,
                provider=self.name,
                model_id=model.id,
            ) from exc


__all__ = ["OpenAICompatibleProvider"]
