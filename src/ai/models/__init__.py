"""models — composable model layer.

Usage::

    import ai
    from ai.types import Message, TextPart

    # look up a model from the catalog
    opus = ai.model("ai-gateway", "anthropic/claude-opus-4-6")

    msgs = [Message(role="user", parts=[TextPart(text="hello")])]

    # stream — auto-creates client from env vars
    s = await ai.stream(opus, msgs)
    async for msg in s:
        print(msg.text_delta, end="")

    # explicit client
    client = ai.Client(
        base_url="https://custom.example.com/v3/ai", api_key="sk-...",
    )
    s = await ai.stream(opus, msgs, client=client)
    async for msg in s:
        ...
"""

from ..types.stream import StreamResultLike
from .ai_gateway.types import GenerateParams, ImageParams, VideoParams
from .core.adapters import register_check, register_generate, register_stream
from .core.api import check_connection, generate, stream
from .core.catalog import get_models, get_providers, register_catalog
from .core.catalog import model as model
from .core.client import _PROVIDER_DEFAULTS, Client
from .core.model import Model, ModelCost
from .core.proto import CheckConnFn, GenerateFn, StreamFn
from .core.stream_result import StreamResult

__all__ = [
    # Core types
    "CheckConnFn",
    "Client",
    "GenerateFn",
    "GenerateParams",
    "ImageParams",
    "Model",
    "ModelCost",
    "StreamFn",
    "StreamResult",
    "StreamResultLike",
    "VideoParams",
    # Catalog
    "get_models",
    "get_providers",
    "model",
    "register_catalog",
    # Adapter / check registration
    "register_check",
    "register_generate",
    "register_stream",
    # Public API
    "check_connection",
    "generate",
    "stream",
    # Internal (used by tests)
    "_PROVIDER_DEFAULTS",
]
