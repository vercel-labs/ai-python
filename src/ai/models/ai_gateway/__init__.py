"""AI Gateway provider.

Usage::

    from ai.models import ai_gateway, anthropic, openai

    model = ai_gateway("anthropic/claude-sonnet-4")
    ids = await ai_gateway.list()

    # Provider-specific request options pass through as raw Gateway body fields.
    async with ai.stream(
        model,
        msgs,
        params={"providerOptions": {"anthropic": {"speed": "fast"}}},
        tools=[anthropic.tools.web_search(max_uses=5)],
    ) as s:
        ...

    # The gateway also exposes its own provider-executed tools that work
    # with any gateway-routed model regardless of the underlying provider.
    async with ai.stream(
        model,
        msgs,
        tools=[ai_gateway.tools.perplexity_search(max_results=5)],
    ) as s:
        ...

The heavy ``.adapter`` module is loaded lazily so that ``import ai`` does
not pull in ``httpx`` and other I/O libraries at import time.  This matters
for sandboxed runtimes (e.g. Temporal workflow workers).
"""

from . import errors, params, tools
from .params import (
    AutoToolChoice,
    GatewayFunctionToolArgs,
    LanguageParams,
    NamedToolChoice,
    NoneToolChoice,
    RequiredToolChoice,
)
from .provider import ai_gateway

__all__ = [
    "GatewayFunctionToolArgs",
    "LanguageParams",
    "AutoToolChoice",
    "NamedToolChoice",
    "NoneToolChoice",
    "RequiredToolChoice",
    "ai_gateway",
    "errors",
    "params",
    "tools",
]


def __getattr__(name: str) -> object:
    if name == "generate":
        from .adapter import generate

        return generate
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
