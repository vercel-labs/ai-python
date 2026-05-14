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

from ... import types
from ...providers.base import Provider as Provider

if TYPE_CHECKING:
    from .model import Model

__all__ = ["GenerateFn", "Provider", "StreamFn"]


@runtime_checkable
class StreamFn(Protocol):
    """Protocol for streaming adapter functions.

    Implementations yield event objects as the response streams in. The
    terminal assistant state is surfaced as a ``StreamEnd.message``.
    """

    def __call__(
        self,
        model: Model,
        messages: list[types.messages.Message],
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[types.events.Event]: ...


@runtime_checkable
class GenerateFn(Protocol):
    """Protocol for non-streaming adapter functions (images, video, etc.).

    ``params`` is typed as ``Any`` at the protocol level because each adapter
    defines its own parameter types (e.g. ``ImageParams | VideoParams``).
    Type safety is enforced at the top-level ``generate()`` function.
    """

    async def __call__(
        self,
        model: Model,
        messages: list[types.messages.Message],
        params: Any,
    ) -> types.messages.Message: ...
