"""OpenAI provider.

Usage::

    from ai.models import openai

    model = openai("gpt-5.4")
    ids = await openai.list()

The adapter module is loaded lazily to avoid pulling in the ``openai``
SDK at import time.
"""

from . import tools
from .provider import openai

__all__ = ["openai", "tools"]


def __getattr__(name: str) -> object:
    if name == "stream":
        from .adapter import stream

        return stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
