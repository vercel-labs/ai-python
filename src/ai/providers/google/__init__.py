"""Google provider.

Usage::

    import ai
    from ai.providers.google import tools as google_tools

    model = ai.get_model("google:gemini-2.5-flash")
    provider = ai.get_provider("google", api_key="...")
    model = ai.Model(id="gemini-2.5-flash", provider=provider)
    ids = await ai.get_provider("google").list_models()

    # built-in tools
    async with ai.stream(
        model, msgs,
        tools=[google_tools.google_search()],
    ) as s:
        ...

The optional upstream ``google-genai`` SDK is loaded lazily when the
provider creates or uses an SDK client.
"""

from . import tools
from .protocol import GoogleGenerateContentProtocol
from .provider import GoogleProvider

__all__ = ["GoogleGenerateContentProtocol", "GoogleProvider", "tools"]
