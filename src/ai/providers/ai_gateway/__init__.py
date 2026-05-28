"""AI Gateway provider.

Usage::

    import ai
    from ai.providers.ai_gateway import tools as gateway_tools
    from ai.providers.anthropic import tools as anthropic_tools

    model = ai.get_model("gateway:anthropic/claude-sonnet-4")
    ids = await ai.get_provider("vercel").list_models()

    # Raw provider-specific request options pass through in extra_body.
    async with ai.stream(
        model,
        msgs,
        params=ai.InferenceRequestParams(
            extra_body={"providerOptions": {"anthropic": {"speed": "fast"}}}
        ),
        tools=[anthropic_tools.web_search(max_uses=5)],
    ) as s:
        ...

    # The gateway also exposes its own provider-executed tools that work
    # with any gateway-routed model regardless of the underlying provider.
    async with ai.stream(
        model,
        msgs,
        tools=[gateway_tools.perplexity_search(max_results=5)],
    ) as s:
        ...

"""

from . import errors, tools
from .params import GatewayParams, ProviderTimeoutsParams
from .protocol import GatewayV3Protocol
from .provider import GatewayProvider

__all__ = [
    "GatewayParams",
    "GatewayProvider",
    "GatewayV3Protocol",
    "ProviderTimeoutsParams",
    "errors",
    "tools",
]
