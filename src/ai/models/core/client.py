"""HTTP client for adapter functions."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

    from . import model as model_


@dataclasses.dataclass
class Client:
    """Connection parameters for a provider API.

    Adapter functions receive a ``Client`` instead of creating their own HTTP
    session.  This keeps auth and base URL decoupled from the adapter logic.

    The :pyattr:`http` property lazily creates a shared
    :class:`httpx.AsyncClient` so that consecutive calls reuse the same
    connection pool.
    """

    base_url: str
    api_key: str | None = None
    headers: dict[str, str] = dataclasses.field(default_factory=dict)

    _http: httpx.AsyncClient | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    @property
    def http(self) -> httpx.AsyncClient:
        """Lazy-init shared httpx client."""
        import httpx as _httpx

        if self._http is None or self._http.is_closed:
            self._http = _httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=_httpx.Timeout(timeout=300.0, connect=10.0),
            )
        return self._http

    async def aclose(self) -> None:
        """Close the underlying HTTP client if open."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
            self._http = None


# ---------------------------------------------------------------------------
# Provider defaults — base URLs and env var names for auto-client creation.
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "ai-gateway": ("https://ai-gateway.vercel.sh/v3/ai", "AI_GATEWAY_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def auto_client(model: model_.Model) -> Client:
    """Create a :class:`Client` from env vars for the given model's provider."""
    defaults = _PROVIDER_DEFAULTS.get(model.provider)
    if defaults is None:
        raise ValueError(
            f"No default client config for provider {model.provider!r}. "
            f"Pass an explicit client= argument."
        )
    base_url, env_var = defaults
    return Client(base_url=base_url, api_key=os.environ.get(env_var))
