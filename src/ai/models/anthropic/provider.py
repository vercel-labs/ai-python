"""Anthropic provider.

Defines the callable :data:`anthropic` provider, which satisfies the
:class:`~ai.models.core.proto.Provider` protocol."""

import os
from types import ModuleType

from .. import core

_BASE_URL = "https://api.anthropic.com"
_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
_API_KEY_ENV = "ANTHROPIC_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"


class _Anthropic:
    """Callable provider factory for Anthropic.

    Satisfies the :class:`~ai.models.core.proto.Provider` protocol.
    """

    @property
    def api_key_env(self) -> str:
        return _API_KEY_ENV

    @property
    def base_url(self) -> str:
        return os.environ.get(_BASE_URL_ENV) or _BASE_URL

    @property
    def adapter(self) -> str:
        return "anthropic"

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def tools(self) -> ModuleType:
        """The provider's built-in tool factories.

        Convenience accessor: ``anthropic.tools.web_search(...)``.
        """
        from . import tools as tools_module

        return tools_module

    def client(self) -> core.client.Client:
        """Create a :class:`Client` from env-var credentials.

        ``ANTHROPIC_BASE_URL`` overrides the default base URL when set.
        """
        return core.client.Client(
            base_url=self.base_url,
            api_key=os.environ.get(_API_KEY_ENV),
        )

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`anthropic.check.check`."""
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
        """List available model IDs from the Anthropic API."""
        c = client or self.client()
        headers = {
            "x-api-key": c.api_key or "",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        response = await c.http.get(
            f"{c.base_url.rstrip('/')}/v1/models", headers=headers
        )
        response.raise_for_status()
        data: list[dict[str, object]] = response.json().get("data", [])
        return sorted(str(m["id"]) for m in data)

    def __repr__(self) -> str:
        return "anthropic"


anthropic = _Anthropic()

__all__ = ["anthropic"]
