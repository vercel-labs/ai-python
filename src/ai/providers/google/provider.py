"""Google provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import httpx
import pydantic

from ... import errors as ai_errors
from .. import base
from . import _sdk, errors
from . import protocol as protocol_module
from . import tools as tools_module

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence
    from types import ModuleType

    import google.genai as genai
    import modelsdotdev

    from ...models.core import model as model_
    from ...models.core import params as params_
    from ...types import events
    from ...types import messages as messages_
    from ...types import tools as tools_

    GoogleClient = httpx.AsyncClient | genai.Client
    GoogleSDKClient = genai.Client
else:
    GoogleClient = Any
    GoogleSDKClient = Any

_BASE_URL = "https://generativelanguage.googleapis.com"
_BASE_URL_ENV = "GOOGLE_GEMINI_BASE_URL"
# Alternative API key envs: the first two come from models.dev, the
# last is the SDK's own canonical env var. The first is the primary,
# the rest are fallbacks checked by :attr:`api_key`.
_API_KEY_ENVS = (
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


class GoogleProvider(base.Provider[GoogleSDKClient]):
    """Callable provider for the Google Gemini API."""

    handles: ClassVar[tuple[str, ...]] = ("google", "@ai-sdk/google")

    provider_class_id: Literal["google"] = "google"

    _http_client: httpx.AsyncClient | None = pydantic.PrivateAttr(default=None)
    _close_client_on_aclose: bool = pydantic.PrivateAttr(default=False)
    _has_user_sdk_client: bool = pydantic.PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        self._close_client_on_aclose = True

    def _set_runtime_client(self, client: GoogleClient | None) -> None:
        google_sdk = None
        if client is not None and not isinstance(client, httpx.AsyncClient):
            google_sdk = _sdk.import_sdk(provider=self.name)

        if google_sdk is not None and isinstance(client, google_sdk.Client):
            sdk_client = client
            http_client = None
            self._has_user_sdk_client = True
            self._close_client_on_aclose = False
        elif isinstance(client, httpx.AsyncClient) or client is None:
            sdk_client = None
            http_client = client
            self._has_user_sdk_client = False
            self._close_client_on_aclose = client is None
        else:
            raise TypeError(
                "Google providers require an httpx.AsyncClient or "
                "google.genai.Client"
            )

        self._http_client = http_client
        if sdk_client is not None:
            self._set_client(sdk_client)

    def _make_sdk_client(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> GoogleSDKClient:
        google_sdk = _sdk.import_sdk(provider=self.name)
        http_options: dict[str, Any] = {"base_url": self.base_url}
        if self.headers:
            http_options["headers"] = dict(self.headers)
        if http_client is not None:
            http_options["httpx_async_client"] = http_client
        return google_sdk.Client(
            api_key=self.api_key or None,
            http_options=cast("Any", http_options),
        )

    @property
    def sdk_client(self) -> GoogleSDKClient:
        """Provider SDK client used for Google API requests."""
        return self.client

    @property
    def client(self) -> GoogleSDKClient:
        """Lazily-created SDK client for Google API requests."""
        if self._client is None:
            self._set_client(
                self._make_sdk_client(http_client=self._http_client)
            )
        return super().client

    @property
    def api_key(self) -> str | None:
        """API key configured directly or via one of the Gemini env vars."""
        api_key = super().api_key
        if api_key is not None:
            return api_key
        for env in _API_KEY_ENVS:
            value = self.env.get(env) or os.environ.get(env)
            if value:
                return value
        return None

    def default_protocol(self) -> base.ProviderProtocol[GoogleSDKClient]:
        """Return the default Google generateContent protocol."""
        return protocol_module.GoogleGenerateContentProtocol()

    def is_configured(self) -> bool:
        if self._has_user_sdk_client:
            return True
        if not self.api_key:
            return False
        return all(self._config_value(env) for env in self.config_envs)

    async def aclose(self) -> None:
        """Close the provider-owned SDK client, if any."""
        if self._close_client_on_aclose and self._client is not None:
            await self.client.aio.aclose()

    def stream(
        self,
        model: model_.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
    ) -> AsyncGenerator[events.Event]:
        """Stream via the Google generateContent protocol."""
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
        client: GoogleClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> base.Provider[GoogleSDKClient]:
        resolved_base_url = base_url or base.provider_base_url(
            provider,
            model_provider_config,
        )
        if resolved_base_url is None:
            resolved_base_url = _BASE_URL
        api_key_env, config_envs = base.provider_config(
            provider, model_provider_config
        )
        provider_instance = cls(
            name=provider.id,
            default_base_url=resolved_base_url,
            api_key_value=api_key,
            api_key_env=api_key_env,
            base_url_env=_BASE_URL_ENV if base_url is None else None,
            # models.dev lists alternative API key envs alongside real
            # config envs; the alternates are handled by `api_key`.
            config_envs=tuple(
                config_env
                for config_env in config_envs
                if config_env not in _API_KEY_ENVS
            ),
            headers=dict(headers or {}),
            env=dict(env or {}),
            protocol_override=protocol,
        )
        provider_instance._set_runtime_client(client)
        return provider_instance

    @property
    def tools(self) -> ModuleType:
        """The provider's built-in tool factories.

        Convenience accessor: ``google.tools.google_search()``.
        """
        return tools_module

    async def list_models(self) -> list[str]:
        """List available model IDs from the Google API."""
        genai_errors = _sdk.import_errors(provider=self.name)
        try:
            pager = await self.sdk_client.aio.models.list()
            names = [sdk_model.name async for sdk_model in pager]
        except genai_errors.APIError as exc:
            raise errors.map_error(exc, provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise errors.map_httpx_error(exc, provider=self.name) from exc
        return sorted(name.removeprefix("models/") for name in names if name)

    async def probe(self, model: model_.Model) -> None:
        """Raise unless credentials are valid and the model exists."""
        if not self.is_configured():
            raise ai_errors.ProviderNotConfiguredError(
                f"provider {self.name!r} is not configured",
                provider=self.name,
            )
        genai_errors = _sdk.import_errors(provider=self.name)
        try:
            await self.sdk_client.aio.models.get(model=model.id)
        except genai_errors.APIError as exc:
            raise errors.map_error(
                exc,
                provider=self.name,
                model_id=model.id,
            ) from exc
        except httpx.HTTPError as exc:
            raise errors.map_httpx_error(exc, provider=self.name) from exc


__all__ = ["GoogleProvider"]
