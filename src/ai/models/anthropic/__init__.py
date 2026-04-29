"""Anthropic provider.

Usage::

    from ai.models import anthropic

    model = anthropic("claude-sonnet-4-6")
    ids = await anthropic.list()

The adapter module is loaded lazily to avoid pulling in the ``anthropic``
SDK at import time.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..core import client as client_
from ..core.model import Model

if TYPE_CHECKING:
    pass

_BASE_URL = "https://api.anthropic.com"
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
        return _BASE_URL

    @property
    def adapter(self) -> str:
        return "anthropic"

    @property
    def name(self) -> str:
        return "anthropic"

    def client(self) -> client_.Client:
        """Create a :class:`Client` from env-var credentials."""
        return client_.Client(
            base_url=_BASE_URL,
            api_key=os.environ.get(_API_KEY_ENV),
        )

    async def check(self, client: client_.Client, model: Model) -> bool:
        """Delegate to :func:`anthropic.check.check`."""
        from . import check as check_

        return await check_.check(client, model)

    def __call__(
        self,
        model_id: str,
        *,
        base_url: str | None = None,
        client: client_.Client | None = None,
    ) -> Model:
        return Model(
            id=model_id,
            adapter=self.adapter,
            provider=self,
            client=client,
        )

    async def list(self, *, client: client_.Client | None = None) -> list[str]:
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


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
