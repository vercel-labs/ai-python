"""Base provider implementation."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.core.client import Client
    from ..models.core.model import Model


class Provider:
    """Base class for model providers.

    A provider carries all provider-specific configuration and behaviour:
    API endpoint, authentication, client creation, connection checks, and
    model enumeration. Model objects hold only pure metadata (``id``,
    ``adapter``) plus a back-reference to their provider.

    Implementations must be **callable** — ``provider(model_id)`` returns
    a :class:`Model`.
    """

    def __init__(
        self,
        *,
        name: str,
        adapter: str,
        base_url: str,
        api_key_env: str | None = None,
        base_url_env: str | None = None,
        config_envs: Iterable[str] | None = None,
    ) -> None:
        self._name = name
        self._adapter = adapter
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._base_url_env = base_url_env
        self._config_envs = tuple(config_envs or ())

    @property
    def api_key_env(self) -> str | None:
        """Env var name that holds the API key (e.g. ``"OPENAI_API_KEY"``)."""
        return self._api_key_env

    @property
    def base_url_env(self) -> str | None:
        """Env var name that can override the default base URL."""
        return self._base_url_env

    @property
    def default_base_url(self) -> str:
        """Base URL configured on the provider before env overrides."""
        return self._base_url

    @property
    def base_url(self) -> str:
        """Default base URL for the provider API."""
        if self._base_url_env:
            base_url = os.environ.get(self._base_url_env) or self._base_url
        else:
            base_url = self._base_url
        for env in self._config_envs:
            value = os.environ.get(env)
            if value is not None:
                base_url = base_url.replace(f"${{{env}}}", value)
                base_url = base_url.replace(f"${env}", value)
        return base_url

    @property
    def adapter(self) -> str:
        """Wire-protocol key used to look up stream/generate adapters."""
        return self._adapter

    @property
    def config_envs(self) -> tuple[str, ...]:
        """Additional env vars used to configure the provider client."""
        return self._config_envs

    @property
    def name(self) -> str:
        """Human-readable provider name (for repr, error messages)."""
        return self._name

    def client(self) -> Client:
        """Create a :class:`Client` from the provider's default config.

        Reads ``api_key_env`` from the environment and uses ``base_url``
        as the endpoint.
        """
        from ..models.core import client as client_

        return client_.Client(
            base_url=self.base_url,
            api_key=os.environ.get(self.api_key_env) if self.api_key_env else None,
        )

    async def check(self, client: Client, model: Model) -> bool:
        """Check whether *client* can reach this provider and *model* exists.

        Returns ``True`` when credentials are valid **and** the model is
        available. Non-auth transport errors should be raised.
        """
        raise NotImplementedError

    async def list(self, *, client: Client | None = None) -> list[str]:
        """List available model IDs from the provider API."""
        raise NotImplementedError

    def __call__(
        self,
        model_id: str,
        *,
        client: Client | None = None,
    ) -> Model:
        """Create a :class:`Model` for the given *model_id*."""
        from ..models.core import model as model_

        return model_.Model(
            id=model_id,
            adapter=self.adapter,
            provider=self,
            client=client,
        )

    def __repr__(self) -> str:
        return self.name
