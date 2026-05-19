"""AI Gateway provider.

Defines the callable :data:`ai_gateway` provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ... import errors as ai_errors
from .. import base
from . import client as gateway_client
from . import errors
from . import protocol as protocol_module
from .client import errors as client_errors

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence
    from types import ModuleType

    import httpx
    import modelsdotdev
    import pydantic

    from ...models.core import model as model_
    from ...models.core import params as params_
    from ...types import events
    from ...types import messages as messages_
    from ...types import tools as tools_

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
        headers: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> None:
        super().__init__(
            name="ai-gateway",
            base_url=base_url,
            protocol=protocol or protocol_module.GatewayV3Protocol(),
            api_key=api_key,
            api_key_env=_API_KEY_ENV,
            headers=headers,
            env=env,
        )
        self._set_client(
            gateway_client.GatewayClient(
                base_url=self.base_url,
                api_key=self.api_key,
                headers=self.headers,
                client=client,
            )
        )

    @property
    def client(self) -> gateway_client.GatewayClient:
        client = super().client
        client.base_url = self.base_url
        client.api_key = self.api_key
        client.headers = self.headers
        return client

    async def aclose(self) -> None:
        """Close the provider-owned Gateway client, if any."""
        await self.client.aclose()

    def stream(
        self,
        model: model_.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[tools_.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: Any = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> AsyncGenerator[events.Event]:
        """Stream via the AI Gateway v3 protocol."""
        return super().stream(
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
            protocol=protocol,
        )

    async def generate(
        self,
        model: model_.Model,
        messages: list[messages_.Message],
        params: params_.GenerateParams,
        *,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> messages_.Message:
        """Generate media via the AI Gateway v3 protocol."""
        return await super().generate(
            model, messages, params, protocol=protocol
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
        client: httpx.AsyncClient | None = None,
        protocol: base.ProviderProtocol[Any] | None = None,
    ) -> base.Provider[gateway_client.GatewayClient]:
        return cls(
            api_key=api_key,
            base_url=base_url or _BASE_URL,
            headers=headers,
            env=env,
            client=client,
            protocol=protocol,
        )

    @property
    def tools(self) -> ModuleType:
        """Gateway-native built-in tool factories.

        Convenience accessor: ``ai_gateway.tools.perplexity_search(...)``.
        These tools are executed server-side by the gateway and work
        with any gateway-routed model.
        """
        from . import tools as tools_module  # noqa: PLC0415

        return tools_module

    async def list_models(self) -> list[str]:
        """List available model IDs from the AI Gateway."""
        try:
            return await self.client.list_model_ids()
        except client_errors.GatewayError as exc:
            raise errors.map_error(exc) from exc

    async def probe(self, model: model_.Model) -> None:
        """Raise unless gateway credentials are valid and the model exists."""
        if not self.is_configured():
            raise ai_errors.ProviderNotConfiguredError(
                f"provider {self.name!r} is not configured",
                provider=self.name,
            )

        try:
            await self.client.probe_model(model.id)
        except client_errors.GatewayError as exc:
            raise errors.map_error(exc) from exc


__all__ = ["GatewayProvider"]
