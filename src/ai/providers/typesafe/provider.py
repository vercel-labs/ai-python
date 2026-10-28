"""TypeSafe provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import pydantic

from ... import errors as ai_errors
from .. import base
from . import _sdk, errors
from . import protocol as protocol_module

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx2
    import modelsdotdev
    import typesafe_sdk

    from ...models.core import model as model_

    TypeSafeClient = typesafe_sdk.AsyncTypeSafeClient
    TypeSafeRuntimeClient = (
        typesafe_sdk.AsyncTypeSafeClient | httpx2.AsyncClient
    )
else:
    TypeSafeClient = Any
    TypeSafeRuntimeClient = Any

_BASE_URL = "https://api.typesafe.ai"
_BASE_URL_ENV = "TYPESAFE_BASE_URL"
_API_KEY_ENV = "TYPESAFE_API_KEY"


class TypeSafeProvider(base.Provider[TypeSafeClient]):
    """Provider for the TypeSafe System One API.

    ``client`` may be a ``typesafe_sdk.AsyncTypeSafeClient``, used as is, or
    an ``httpx2.AsyncClient`` to send requests through. Provider-created SDK
    clients are closed by :meth:`aclose`; user-supplied clients are not.
    """

    handles: ClassVar[tuple[str, ...]] = ("typesafe",)

    provider_class_id: Literal["typesafe"] = "typesafe"
    name: str = "typesafe"
    default_base_url: str = _BASE_URL
    api_key_env: str | None = _API_KEY_ENV
    base_url_env: str | None = _BASE_URL_ENV

    _http_client: httpx2.AsyncClient | None = pydantic.PrivateAttr(default=None)
    _has_user_sdk_client: bool = pydantic.PrivateAttr(default=False)

    def __init__(
        self,
        *,
        client: TypeSafeRuntimeClient | None = None,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        if client is None:
            return
        typesafe = _sdk.import_sdk(provider=self.name)
        if isinstance(client, typesafe.AsyncTypeSafeClient):
            self._has_user_sdk_client = True
            self._set_client(client)
        else:
            self._http_client = cast("httpx2.AsyncClient", client)

    @property
    def client(self) -> TypeSafeClient:
        """Lazily-created TypeSafe SDK client."""
        if self._client is None:
            # The SDK client rejects a missing API key at construction.
            api_key = self.api_key
            if not api_key:
                raise ai_errors.ProviderNotConfiguredError(
                    f"provider {self.name!r} is not configured: set "
                    f"{self.api_key_env or 'an API key'}",
                    provider=self.name,
                )
            typesafe = _sdk.import_sdk(provider=self.name)
            self._set_client(
                typesafe.AsyncTypeSafeClient(
                    api_key=api_key,
                    base_url=self.base_url,
                    headers=dict(self.headers),
                    http_client=self._http_client,
                )
            )
        return super().client

    def default_protocol(self) -> base.ProviderProtocol[TypeSafeClient]:
        """Return the TypeSafe System One protocol."""
        return protocol_module.TypeSafeSystemOneProtocol()

    def is_configured(self) -> bool:
        if self._has_user_sdk_client:
            return True
        return super().is_configured()

    async def aclose(self) -> None:
        """Close the provider-created SDK client, if any."""
        # Closing the SDK client also closes its HTTP client, so leave
        # clients built around a user-supplied HTTP client open.
        if (
            self._client is not None
            and not self._has_user_sdk_client
            and self._http_client is None
        ):
            client, self._client = self._client, None
            await client.aclose()

    async def list_models(self) -> list[str]:
        """List model names and aliases from the TypeSafe API."""
        typesafe = _sdk.import_sdk(provider=self.name)
        try:
            response = await self.client.models.list()
        except typesafe.TypeSafeError as exc:
            raise errors.map_error(exc, provider=self.name) from exc
        return sorted(model.name for model in response.models)

    async def probe(self, model: model_.Model) -> None:
        """Raise unless credentials are valid and the model is listed.

        TypeSafe lists aliases such as ``jev-latest``; pinned versions such
        as ``jev-1.13.0`` are accepted by the API but are not listed.
        """
        if not self.is_configured():
            raise ai_errors.ProviderNotConfiguredError(
                f"provider {self.name!r} is not configured",
                provider=self.name,
            )
        if model.id not in await self.list_models():
            raise ai_errors.ProviderModelNotFoundError(
                f"model {model.id!r} is not listed by {self.name!r}",
                model_id=model.id,
                provider=self.name,
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
        client: TypeSafeRuntimeClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> base.Provider[TypeSafeClient]:
        api_key_env, config_envs = base.provider_config(
            provider, model_provider_config
        )
        resolved_base_url = base_url or base.provider_base_url(
            provider, model_provider_config
        )
        return cls(
            name=provider.id,
            default_base_url=resolved_base_url or _BASE_URL,
            base_url_env=None if base_url else _BASE_URL_ENV,
            api_key_value=api_key,
            api_key_env=api_key_env or _API_KEY_ENV,
            config_envs=config_envs,
            headers=dict(headers or {}),
            env=dict(env or {}),
            protocol_override=protocol,
            client=client,
        )


__all__ = ["TypeSafeProvider"]
