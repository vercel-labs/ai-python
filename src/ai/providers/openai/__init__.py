"""OpenAI provider.

Usage::

    import ai

    model = ai.get_model("openai:gpt-5.4")
    model = ai.Model(
        "llama3",
        provider=ai.ProviderRef(
            "openai",
            base_url="http://localhost:11434/v1",
        ),
    )
    ids = await ai.get_provider("openai").list_models()

The optional upstream OpenAI SDK is loaded lazily when the provider creates or
uses an SDK client.
"""

from . import tools
from .protocol import OpenAIChatCompletionsProtocol, OpenAIResponsesProtocol
from .provider import OpenAICompatibleProvider

__all__ = [
    "OpenAIChatCompletionsProtocol",
    "OpenAICompatibleProvider",
    "OpenAIResponsesProtocol",
    "tools",
]
