"""Anthropic provider.

Usage::

    from ai.models import anthropic

    model = anthropic("claude-sonnet-4-6")
    ids = await anthropic.list()

    # built-in tools
    async with ai.stream(
        model, msgs,
        tools=[anthropic.tools.web_search(max_uses=5)],
    ) as s:
        ...

The adapter module is loaded lazily to avoid pulling in the ``anthropic``
SDK at import time.
"""

from . import tools
from .provider import anthropic

__all__ = ["anthropic", "tools"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
