"""Anthropic provider.

Usage::

    import ai
    from ai.providers.anthropic import tools as anthropic_tools

    model = ai.get_model("anthropic:claude-sonnet-4-6")
    provider = ai.get_provider("anthropic", base_url="https://anthropic.example.com")
    model = ai.Model("claude-sonnet-4-6", provider=provider)
    ids = await ai.get_provider("anthropic").list()

    # built-in tools
    async with ai.stream(
        model, msgs,
        tools=[anthropic_tools.web_search(max_uses=5)],
    ) as s:
        ...

The adapter module is loaded lazily to avoid pulling in the ``anthropic``
SDK at import time.
"""

from . import tools
from .provider import AnthropicCompatibleProvider

__all__ = ["AnthropicCompatibleProvider", "tools"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
