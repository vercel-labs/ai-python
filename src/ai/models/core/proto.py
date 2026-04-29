"""Adapter function protocols.

An *adapter function* translates between our ``Message`` types and a specific
provider API (e.g. ``"ai-gateway-v3"``, ``"anthropic-messages"``).

Adapter functions are plain async generators / coroutines — no base class
required.  The protocols below exist only for static type-checking.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pydantic

from ...types import events as events_
from ...types import messages as messages_
from ...types import proto as types_proto_

if TYPE_CHECKING:
    from .client import Client
    from .model import Model


@runtime_checkable
class Provider(Protocol):
    """Protocol for model providers.

    A provider carries all provider-specific configuration and behaviour:
    API endpoint, authentication, client creation, connection checks, and
    model enumeration.  Model objects hold only pure metadata (``id``,
    ``adapter``) plus a back-reference to their provider.

    Implementations must be **callable** — ``provider(model_id)`` returns
    a :class:`Model`.
    """

    @property
    def api_key_env(self) -> str | None:
        """Env var name that holds the API key (e.g. ``"OPENAI_API_KEY"``)."""
        ...

    @property
    def base_url(self) -> str:
        """Default base URL for the provider API."""
        ...

    @property
    def adapter(self) -> str:
        """Wire-protocol key used to look up stream/generate adapters."""
        ...

    @property
    def name(self) -> str:
        """Human-readable provider name (for repr, error messages)."""
        ...

    def client(self) -> Client:
        """Create a :class:`Client` from the provider's default config.

        Reads ``api_key_env`` from the environment and uses ``base_url``
        as the endpoint.
        """
        ...

    async def check(self, client: Client, model: Model) -> bool:
        """Check whether *client* can reach this provider and *model* exists.

        Returns ``True`` when credentials are valid **and** the model is
        available.  Non-auth transport errors should be raised.
        """
        ...

    async def list(self, *, client: Client | None = None) -> list[str]:
        """List available model IDs from the provider API."""
        ...

    def __call__(
        self,
        model_id: str,
        *,
        client: Client | None = None,
    ) -> Model:
        """Create a :class:`Model` for the given *model_id*."""
        ...


@runtime_checkable
class StreamFn(Protocol):
    """Protocol for streaming adapter functions.

    Implementations yield event objects as the response streams in. The
    terminal assistant state is surfaced as a ``StreamEnd.message``.
    """

    def __call__(
        self,
        client: Client,
        model: Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[types_proto_.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]: ...


@runtime_checkable
class GenerateFn(Protocol):
    """Protocol for non-streaming adapter functions (images, video, etc.).

    ``params`` is typed as ``Any`` at the protocol level because each adapter
    defines its own parameter types (e.g. ``ImageParams | VideoParams``).
    Type safety is enforced at the top-level ``generate()`` function.
    """

    async def __call__(
        self,
        client: Client,
        model: Model,
        messages: list[messages_.Message],
        params: Any,
    ) -> messages_.Message: ...


@runtime_checkable
class CheckConnFn(Protocol):
    """Protocol for connection-check functions.

    A check function verifies that *client* can reach the provider and that
    *model* is available there.  Returns ``True`` when the credentials are
    valid **and** the model exists on the remote side.

    The check must be **free** — it should only hit metadata / listing
    endpoints that don't consume tokens or credits.

    Non-auth transport errors (network failures, 5xx) should be raised
    rather than returning ``False`` so that callers can distinguish
    "bad credentials" from "provider unreachable".
    """

    async def __call__(
        self,
        client: Client,
        model: Model,
    ) -> bool: ...
