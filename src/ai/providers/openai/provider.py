"""OpenAI-compatible providers."""

from collections.abc import Iterable
from types import ModuleType

from ...models import core
from .. import base

_BASE_URL = "https://api.openai.com/v1"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAICompatibleProvider(base.Provider):
    """Callable provider for OpenAI-compatible APIs.

    ``provider("gpt-5.4")`` returns a :class:`Model` that uses the OpenAI
    chat-completions adapter.
    """

    def __init__(
        self,
        *,
        name: str,
        default_base_url: str,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        config_envs: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            adapter="openai",
            base_url=default_base_url,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            config_envs=config_envs,
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

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`openai.check.check`."""
        from . import check as check_

        return await check_.check(client, model)

    async def list(self, *, client: core.client.Client | None = None) -> list[str]:
        """List available model IDs from the OpenAI-compatible API."""
        c = client or self.client()
        headers = {"Authorization": f"Bearer {c.api_key}"} if c.api_key else {}
        response = await c.http.get(f"{c.base_url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        data: list[dict[str, object]] = response.json().get("data", [])
        return sorted(str(m["id"]) for m in data)


def openai_like(
    *,
    name: str,
    base_url: str,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
    config_envs: Iterable[str] | None = None,
) -> OpenAICompatibleProvider:
    """Create a provider for an OpenAI-compatible API."""
    return OpenAICompatibleProvider(
        name=name,
        default_base_url=base_url,
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        config_envs=config_envs,
    )


openai = openai_like(
    name="openai",
    base_url=_BASE_URL,
    api_key_env=_API_KEY_ENV,
    base_url_env=_BASE_URL_ENV,
)

__all__ = ["OpenAICompatibleProvider", "openai", "openai_like"]
