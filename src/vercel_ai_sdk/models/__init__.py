"""models — composable model layer.

Usage::

    from vercel_ai_sdk import models
    from vercel_ai_sdk.types import Message, TextPart

    model = models.Model(
        id="anthropic/claude-sonnet-4",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )
    msgs = [Message(role="user", parts=[TextPart(text="hello")])]

    # stream — auto-creates client from env vars
    s = await models.stream(model, msgs)
    async for msg in s:
        print(msg.text_delta, end="")

    # buffer the whole response
    result = await models.buffer(await models.stream(model, msgs))
    print(result.text)

    # explicit client
    client = models.Client(
        base_url="https://custom.example.com/v3/ai", api_key="sk-...",
    )
    s = await models.stream(model, msgs, client=client)
    async for msg in s:
        ...
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

from ..types import messages as messages_
from ..types import tools as tools_
from .ai_gateway.generate import GenerateParams, ImageParams, VideoParams
from .core.client import Client
from .core.model import Model, ModelCost
from .core.proto import GenerateFn, StreamFn

# ---------------------------------------------------------------------------
# Adapter registry — maps adapter string → adapter function.
# Adapter modules are imported lazily on first use.
# ---------------------------------------------------------------------------

_stream_adapters: dict[str, StreamFn] = {}
_generate_adapters: dict[str, GenerateFn] = {}
_adapters_loaded = False


def _ensure_adapters() -> None:
    """Lazily register built-in adapter functions on first call."""
    global _adapters_loaded  # noqa: PLW0603
    if _adapters_loaded:
        return
    _adapters_loaded = True

    from .ai_gateway import generate as ai_gw_generate
    from .ai_gateway import stream as ai_gw_stream
    from .anthropic.adapter import stream as anthropic_stream
    from .openai.adapter import stream as openai_stream

    _stream_adapters["ai-gateway-v3"] = ai_gw_stream
    _generate_adapters["ai-gateway-v3"] = ai_gw_generate
    _stream_adapters["openai"] = openai_stream
    _stream_adapters["anthropic"] = anthropic_stream


def register_stream(adapter: str, fn: StreamFn) -> None:
    """Register a stream adapter function for the given adapter key.

    Use this to add custom adapters (or override built-in ones).
    """
    _stream_adapters[adapter] = fn


def register_generate(adapter: str, fn: GenerateFn) -> None:
    """Register a generate adapter function for the given adapter key.

    Use this to add custom adapters (or override built-in ones).
    """
    _generate_adapters[adapter] = fn


# ---------------------------------------------------------------------------
# Provider defaults — base URLs and env var names for auto-client creation.
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "ai-gateway": ("https://ai-gateway.vercel.sh/v3/ai", "AI_GATEWAY_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def _auto_client(model: Model) -> Client:
    """Create a :class:`Client` from env vars for the given model's provider."""
    defaults = _PROVIDER_DEFAULTS.get(model.provider)
    if defaults is None:
        raise ValueError(
            f"No default client config for provider {model.provider!r}. "
            f"Pass an explicit client= argument."
        )
    base_url, env_var = defaults
    return Client(base_url=base_url, api_key=os.environ.get(env_var))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StreamResult:
    """Wrapper around a message stream. Async-iterable; collects the final result.

    Properties like ``.text`` and ``.tool_calls`` delegate to the final
    ``Message`` snapshot and are available after iteration completes.
    """

    def __init__(self, gen: AsyncGenerator[messages_.Message]) -> None:
        self._gen = gen
        self._final: messages_.Message | None = None

    def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        return self._iterate()

    async def _iterate(self) -> AsyncGenerator[messages_.Message]:
        async for msg in self._gen:
            self._final = msg
            yield msg

    @property
    def text(self) -> str:
        return self._final.text if self._final else ""

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]:
        return self._final.tool_calls if self._final else []

    @property
    def usage(self) -> messages_.Usage | None:
        return self._final.usage if self._final else None

    @property
    def output(self) -> Any:
        """Parsed structured output from the final message, if available."""
        return self._final.output if self._final else None


async def stream(
    model: Model,
    messages: list[messages_.Message],
    *,
    tools: Sequence[tools_.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    client: Client | None = None,
    **kwargs: Any,
) -> StreamResult:
    """Stream an LLM response.

    Returns a :class:`StreamResult` that is async-iterable and collects
    the final ``Message``.  After iteration, access ``.text``,
    ``.tool_calls``, ``.usage``, etc.

    If a :class:`~vercel_ai_sdk.agents3.durability.DurabilityProvider` is
    active (set by ``Agent.run()``), the stream is routed through the
    provider for recording or replay.
    """
    # Lazy import to avoid circular dependency at module level.
    from .._durability import get_provider

    provider = get_provider()

    async def _make_raw() -> StreamResult:
        _ensure_adapters()
        c = client or _auto_client(model)
        adapter_fn = _stream_adapters.get(model.adapter)
        if adapter_fn is None:
            registered = ", ".join(sorted(_stream_adapters)) or "(none)"
            raise KeyError(
                f"No stream adapter registered for adapter={model.adapter!r}. "
                f"Registered: {registered}"
            )
        return StreamResult(
            adapter_fn(
                c, model, messages, tools=tools, output_type=output_type, **kwargs
            )
        )

    if provider is not None:
        # Provider returns a StreamResultLike — may be a replay or
        # a recording wrapper.  We return it typed as StreamResult;
        # callers only use the shared protocol surface (.tool_calls, etc.).
        return await provider.execute_stream(_make_raw)  # type: ignore[return-value]

    return await _make_raw()


async def generate(
    model: Model,
    messages: list[messages_.Message],
    params: GenerateParams | None = None,
    *,
    client: Client | None = None,
) -> messages_.Message:
    """Generate a response (images, video, etc.).

    Resolves the adapter function from ``model.adapter``, auto-creates a
    :class:`Client` from env vars if none is provided.

    ``params`` controls the generation type:

    * :class:`ImageParams` — image generation (``/image-model``).
    * :class:`VideoParams` — video generation (``/video-model``).
    * ``None`` — auto-detect from ``model.capabilities``.
    """
    _ensure_adapters()
    c = client or _auto_client(model)
    adapter_fn = _generate_adapters.get(model.adapter)
    if adapter_fn is None:
        registered = ", ".join(sorted(_generate_adapters)) or "(none)"
        raise KeyError(
            f"No generate adapter registered for adapter={model.adapter!r}. "
            f"Registered: {registered}"
        )
    return await adapter_fn(c, model, messages, params=params)


async def buffer(gen: AsyncGenerator[messages_.Message]) -> messages_.Message:
    """Drain a stream and return the final ``Message``.

    Raises :class:`ValueError` if the stream yields nothing.
    """
    result: messages_.Message | None = None
    async for msg in gen:
        result = msg
    if result is None:
        raise ValueError("empty stream")
    return result


__all__ = [
    # Core types
    "Client",
    "GenerateFn",
    "GenerateParams",
    "ImageParams",
    "Model",
    "ModelCost",
    "StreamFn",
    "StreamResult",
    "VideoParams",
    # Public API
    "buffer",
    "generate",
    "register_generate",
    "register_stream",
    "stream",
]
