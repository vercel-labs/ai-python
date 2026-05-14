"""OpenAI-compatible providers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar

import httpx
import openai

from ... import errors as ai_errors
from .. import base
from . import errors

if TYPE_CHECKING:
    import modelsdotdev

    from ...models.core import model as model_

OpenAIClient = httpx.AsyncClient | openai.AsyncOpenAI

_BASE_URL = "https://api.openai.com/v1"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAICompatibleProvider(base.Provider[openai.AsyncOpenAI]):
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
    ) -> None:
        if isinstance(client, openai.AsyncOpenAI):
            sdk_client = client
            http_client = None
            self._has_user_sdk_client = True
        elif isinstance(client, httpx.AsyncClient) or client is None:
            sdk_client = None
            http_client = client
            self._has_user_sdk_client = False
        else:
            raise TypeError(
                "OpenAI providers require an httpx.AsyncClient or openai.AsyncOpenAI"
            )

        super().__init__(
            name=name,
            adapter="openai",
            base_url=default_base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            config_envs=config_envs,
            headers=headers,
            env=env,
        )
        self._close_client_on_aclose = sdk_client is None and http_client is None
        if sdk_client is None:
            sdk_client = self._make_sdk_client(http_client=http_client)
        self._set_client(sdk_client)

    def _make_sdk_client(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "",
            default_headers=self.headers,
            http_client=http_client,
        )

    @property
    def sdk_client(self) -> openai.AsyncOpenAI:
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
        if self._close_client_on_aclose:
            await self.client.close()

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
    ) -> base.Provider[openai.AsyncOpenAI]:
        resolved_base_url = base_url or base.provider_base_url(
            provider,
            model_provider_config,
        )
        if resolved_base_url is None and provider.id == "openai":
            resolved_base_url = _BASE_URL
        if resolved_base_url is None:
            raise ValueError(f"provider {provider.id!r} does not declare an API URL")
        api_key_env, config_envs = base.provider_config(provider, model_provider_config)
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
        )

    @property
    def tools(self) -> ModuleType:
        """The provider's built-in tool factories.

        Convenience accessor: ``openai.tools.web_search(...)``. The
        chat-completions adapter currently raises if a built-in tool is
        passed; route via the AI Gateway provider until a Responses
        adapter ships.
        """
        from . import tools as tools_module

        return tools_module

    async def list(self) -> list[str]:
        """List available model IDs from the OpenAI-compatible API."""
        try:
            sdk_models = await self.sdk_client.models.list()
        except openai.OpenAIError as exc:
            raise errors.map_error(exc, provider=self.name) from exc
        return sorted(str(m.id) for m in sdk_models.data)

    async def probe(self, model: model_.Model) -> None:
        """Raise unless credentials are valid and the model exists."""
        if not self.is_configured():
            raise ai_errors.ProviderNotConfiguredError(
                f"provider {self.name!r} is not configured",
                provider=self.name,
            )
        try:
            await self.sdk_client.models.retrieve(model.id)
        except openai.OpenAIError as exc:
            raise errors.map_error(
                exc,
                provider=self.name,
                model_id=model.id,
            ) from exc


__all__ = ["OpenAICompatibleProvider"]
