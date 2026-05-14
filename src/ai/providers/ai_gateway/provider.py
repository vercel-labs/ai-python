"""AI Gateway provider.

Defines the callable :data:`ai_gateway` provider."""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar

import httpx

from ... import errors as ai_errors
from .. import base
from . import client as gateway_client
from . import errors, mapping

if TYPE_CHECKING:
    import modelsdotdev

    from ...models.core import model as model_

_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_API_KEY_ENV = "AI_GATEWAY_API_KEY"


class GatewayProvider(base.Provider[gateway_client.GatewayClient]):
    """Provider configuration for the Vercel AI Gateway."""

    handles: ClassVar[tuple[str, ...]] = ("vercel", "@ai-sdk/gateway")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            name="ai-gateway",
            adapter="ai-gateway-v3",
            base_url=base_url,
            api_key=api_key,
            api_key_env=_API_KEY_ENV,
            env=env,
        )
        self._set_client(
            gateway_client.GatewayClient(
                base_url=self.base_url,
                api_key=self.api_key,
                client=client,
            )
        )

    @property
    def client(self) -> gateway_client.GatewayClient:
        client = super().client
        client.base_url = self.base_url
        client.api_key = self.api_key
        return client

    async def aclose(self) -> None:
        """Close the provider-owned Gateway client, if any."""
        await self.client.aclose()

    @classmethod
    def from_modelsdev_provider(
        cls,
        provider: modelsdotdev.Provider,
        *,
        model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> base.Provider[gateway_client.GatewayClient]:
        return cls(
            api_key=api_key,
            base_url=base_url or _BASE_URL,
            env=env,
            client=client,
        )

    @property
    def tools(self) -> ModuleType:
        """Gateway-native built-in tool factories.

        Convenience accessor: ``ai_gateway.tools.perplexity_search(...)``.
        These tools are executed server-side by the gateway and work
        with any gateway-routed model.
        """
        from . import tools as tools_module

        return tools_module

    async def list(self) -> list[str]:
        """List available model IDs from the AI Gateway."""
        try:
            return await self.client.list_model_ids()
        except errors.GatewayError as exc:
            raise mapping.map_error(exc) from exc

    async def probe(self, model: model_.Model) -> None:
        """Raise unless gateway credentials are valid and the model exists."""
        if not self.is_configured():
            raise ai_errors.ProviderNotConfiguredError(
                f"provider {self.name!r} is not configured",
                provider=self.name,
            )

        try:
            await self.client.probe_model(model.id)
        except errors.GatewayError as exc:
            raise mapping.map_error(exc) from exc


__all__ = ["GatewayProvider"]
