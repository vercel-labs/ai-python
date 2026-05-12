"""Provider implementations and factories."""

from typing import TYPE_CHECKING

from .base import Provider

if TYPE_CHECKING:
    from .ai_gateway import ai_gateway
    from .anthropic import AnthropicCompatibleProvider, anthropic, anthropic_like
    from .openai import OpenAICompatibleProvider, openai, openai_like

_EXPORTS = {
    "AnthropicCompatibleProvider": ".anthropic",
    "OpenAICompatibleProvider": ".openai",
    "ai_gateway": ".ai_gateway",
    "anthropic": ".anthropic",
    "anthropic_like": ".anthropic",
    "openai": ".openai",
    "openai_like": ".openai",
}


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "AnthropicCompatibleProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ai_gateway",
    "anthropic",
    "anthropic_like",
    "openai",
    "openai_like",
]
