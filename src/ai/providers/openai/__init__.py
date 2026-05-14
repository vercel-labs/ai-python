"""OpenAI provider.

Usage::

    import ai

    model = ai.get_model("openai:gpt-5.4")
    provider = ai.get_provider("openai", base_url="http://localhost:11434/v1")
    model = ai.Model("llama3", provider=provider)
    ids = await ai.get_provider("openai").list()

The adapter module is loaded lazily to avoid pulling in the ``openai``
SDK at import time.
"""

from . import tools
from .provider import OpenAICompatibleProvider

__all__ = ["OpenAICompatibleProvider", "tools"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
