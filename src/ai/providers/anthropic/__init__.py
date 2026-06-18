"""Anthropic provider.

Usage::

    import ai
    from ai.providers.anthropic import tools as anthropic_tools

    model = ai.get_model("anthropic:claude-sonnet-4-6")
    provider = ai.get_provider("anthropic", base_url="https://anthropic.example.com")
    model = ai.Model(id="claude-sonnet-4-6", provider=provider)
    ids = await ai.get_provider("anthropic").list_models()

    # built-in tools
    async with ai.stream(
        model, msgs,
        tools=[anthropic_tools.web_search(max_uses=5)],
    ) as s:
        ...

The optional upstream Anthropic SDK is loaded lazily when the provider creates
or uses an SDK client.
"""

from . import tools
from .protocol import AnthropicMessagesProtocol
from .provider import AnthropicCompatibleProvider

__all__ = ["AnthropicCompatibleProvider", "AnthropicMessagesProtocol", "tools"]
