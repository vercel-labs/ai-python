"""Anthropic-compatible providers."""

from collections.abc import Iterable
from types import ModuleType

from ...models import core
from .. import base

_BASE_URL = "https://api.anthropic.com"
_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
_API_KEY_ENV = "ANTHROPIC_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicCompatibleProvider(base.Provider):
    """Callable provider for Anthropic-compatible APIs."""

    def __init__(
        self,
        *,
        name: str,
        default_base_url: str,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        config_envs: Iterable[str] | None = None,
        anthropic_version: str = _ANTHROPIC_VERSION,
    ) -> None:
        super().__init__(
            name=name,
            adapter="anthropic",
            base_url=default_base_url,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            config_envs=config_envs,
        )
        self.anthropic_version = anthropic_version

    @property
    def tools(self) -> ModuleType:
        """The provider's built-in tool factories.

        Convenience accessor: ``anthropic.tools.web_search(...)``.
        """
        from . import tools as tools_module

        return tools_module

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`anthropic.check.check`."""
        from . import check as check_

        return await check_.check(client, model)

    async def list(self, *, client: core.client.Client | None = None) -> list[str]:
        """List available model IDs from the Anthropic API."""
        c = client or self.client()
        headers = {
            "x-api-key": c.api_key or "",
            "anthropic-version": self.anthropic_version,
        }
        response = await c.http.get(
            f"{c.base_url.rstrip('/')}/v1/models", headers=headers
        )
        response.raise_for_status()
        data: list[dict[str, object]] = response.json().get("data", [])
        return sorted(str(m["id"]) for m in data)


def anthropic_like(
    *,
    name: str,
    base_url: str,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
    config_envs: Iterable[str] | None = None,
    anthropic_version: str = _ANTHROPIC_VERSION,
) -> AnthropicCompatibleProvider:
    """Create a provider for an Anthropic-compatible API."""
    return AnthropicCompatibleProvider(
        name=name,
        default_base_url=base_url,
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        config_envs=config_envs,
        anthropic_version=anthropic_version,
    )


anthropic = anthropic_like(
    name="anthropic",
    base_url=_BASE_URL,
    api_key_env=_API_KEY_ENV,
    base_url_env=_BASE_URL_ENV,
)

__all__ = ["AnthropicCompatibleProvider", "anthropic", "anthropic_like"]
