"""OpenAI provider.

Defines the callable :data:`openai` provider, which satisfies the
:class:`~ai.models.core.proto.Provider` protocol."""

import os
from types import ModuleType

from .. import core

_BASE_URL = "https://api.openai.com/v1"
_BASE_URL_ENV = "OPENAI_BASE_URL"
_API_KEY_ENV = "OPENAI_API_KEY"


class _OpenAI:
    """Callable provider — ``openai("gpt-5.4")`` returns a :class:`Model`.

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
        return "openai"

    @property
    def name(self) -> str:
        return "openai"

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

    def client(self) -> core.client.Client:
        """Create a :class:`Client` from env-var credentials.

        ``OPENAI_BASE_URL`` overrides the default base URL when set.
        """
        return core.client.Client(
            base_url=self.base_url,
            api_key=os.environ.get(_API_KEY_ENV),
        )

    async def check(self, client: core.client.Client, model: core.model.Model) -> bool:
        """Delegate to :func:`openai.check.check`."""
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
        """List available model IDs from the OpenAI API."""
        c = client or self.client()
        headers = {"Authorization": f"Bearer {c.api_key}"}
        response = await c.http.get(f"{c.base_url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        data: list[dict[str, object]] = response.json().get("data", [])
        return sorted(str(m["id"]) for m in data)

    def __repr__(self) -> str:
        return "openai"


openai = _OpenAI()

__all__ = ["openai"]
