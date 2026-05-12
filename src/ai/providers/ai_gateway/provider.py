"""AI Gateway provider.

Defines the callable :data:`ai_gateway` provider."""

from types import ModuleType
from typing import Any

from ...models import core
from .. import base

_BASE_URL = "https://ai-gateway.vercel.sh/v3/ai"
_API_KEY_ENV = "AI_GATEWAY_API_KEY"


class _AIGateway(base.Provider):
    """Callable provider factory for the Vercel AI Gateway."""

    def __init__(self) -> None:
        super().__init__(
            name="ai-gateway",
            adapter="ai-gateway-v3",
            base_url=_BASE_URL,
            api_key_env=_API_KEY_ENV,
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

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`ai_gateway.check.check`."""
        from . import check as check_

        return await check_.check(client, model)

    async def list(self, *, client: core.client.Client | None = None) -> list[str]:
        """List available model IDs from the AI Gateway."""
        from . import sdk

        c = client or self.client()
        gateway = sdk.GatewayClient(c)
        response = await gateway.get("config")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return sorted(str(m["id"]) for m in data.get("models", []))


ai_gateway = _AIGateway()

__all__ = ["ai_gateway"]
