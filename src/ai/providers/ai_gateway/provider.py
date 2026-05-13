"""AI Gateway provider.

Defines the callable :data:`ai_gateway` provider."""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from .. import base
from . import client as gateway_client

if TYPE_CHECKING:
    import modelsdotdev

    from ...models.core import model as model_

_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_API_KEY_ENV = "AI_GATEWAY_API_KEY"
_FAIL_STATUSES = frozenset({401, 403})


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
        response = await self.client.get("config")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return sorted(str(m["id"]) for m in data.get("models", []))

    async def check(self, model: model_.Model) -> bool:
        """Return ``True`` when gateway credentials are valid and model exists."""
        if not self.is_configured():
            return False

        auth_resp = await self.client.get("v1/credits", origin=True)
        if auth_resp.status_code in _FAIL_STATUSES:
            return False
        if auth_resp.status_code != 200:
            auth_resp.raise_for_status()

        config_resp = await self.client.get("config")
        if config_resp.status_code != 200:
            config_resp.raise_for_status()
            return False  # pragma: no cover

        data: dict[str, Any] = config_resp.json()
        remote_ids: set[str] = {m["id"] for m in data.get("models", [])}
        return model.id in remote_ids


__all__ = ["GatewayProvider"]
