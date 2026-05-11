"""AI Gateway provider.

Defines the callable :data:`ai_gateway` provider, which satisfies the
:class:`~ai.models.core.proto.Provider` protocol."""

import os
from types import ModuleType
from typing import Any

from .. import core

_BASE_URL = "https://ai-gateway.vercel.sh/v4/ai"
_API_KEY_ENV = "AI_GATEWAY_API_KEY"


class _AIGateway:
    """Callable provider factory for the Vercel AI Gateway.

    Satisfies the :class:`~ai.models.core.proto.Provider` protocol.
    """

    @property
    def api_key_env(self) -> str:
        return _API_KEY_ENV

    @property
    def base_url(self) -> str:
        return _BASE_URL

    @property
    def adapter(self) -> str:
        return "ai-gateway-v4"

    @property
    def name(self) -> str:
        return "ai-gateway"

    @property
    def tools(self) -> ModuleType:
        """Gateway-native built-in tool factories.

        Convenience accessor: ``ai_gateway.tools.perplexity_search(...)``.
        These tools are executed server-side by the gateway and work
        with any gateway-routed model.
        """
        from . import tools as tools_module

        return tools_module

    def client(self) -> core.client.Client:
        """Create a :class:`Client` from env-var credentials."""
        return core.client.Client(
            base_url=_BASE_URL,
            api_key=os.environ.get(_API_KEY_ENV),
        )

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`ai_gateway.check.check`."""
        from . import check as check_

        return await check_.check(client, model)

    def __call__(
        self,
        model_id: str,
        *,
        client: core.client.Client | None = None,
    ) -> core.model.Model:
        return core.model.Model(
            id=model_id,
            adapter=self.adapter,
            provider=self,
            client=client,
        )

    async def list(self, *, client: core.client.Client | None = None) -> list[str]:
        """List available model IDs from the AI Gateway."""
        from . import sdk

        c = client or self.client()
        gateway = sdk.GatewayClient(c)
        response = await gateway.get("config")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return sorted(str(m["id"]) for m in data.get("models", []))

    def __repr__(self) -> str:
        return "ai_gateway"


ai_gateway = _AIGateway()

__all__ = ["ai_gateway"]
