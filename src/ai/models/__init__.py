"""models — composable model layer.

Usage::

    import ai
    from ai.models import openai, anthropic, ai_gateway

    model = openai("gpt-5.4")
    model = anthropic("claude-sonnet-4-6")
    model = ai_gateway("anthropic/claude-sonnet-4")

    # stream — auto-creates client from env vars
    msgs = [ai.user_message("hello")]
    s = ai.stream(model, msgs)
    async for event in s:
        if isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)

    # explicit client for custom auth
    client = ai.Client(base_url="https://custom.example.com/v1", api_key="sk-...")
    model = openai("gpt-5.4", client=client)
    s = ai.stream(model, msgs)

    # list available models
    ids = await openai.list()
"""

from .ai_gateway import ai_gateway
from .anthropic import anthropic
from .core.adapters import register_generate, register_stream
from .core.api import (
    Executor,
    GenerateExecutor,
    GenerateRequest,
    Stream,
    StreamExecutor,
    StreamRequest,
    check_connection,
    generate,
    stream,
)
from .core.client import Client
from .core.model import Model
from .core.params import GenerateParams, ImageParams, VideoParams
from .core.proto import CheckConnFn, GenerateFn, Provider, StreamFn
from .openai import openai

__all__ = [
    # Core types
    "CheckConnFn",
    "Client",
    "Executor",
    "GenerateExecutor",
    "GenerateFn",
    "GenerateParams",
    "GenerateRequest",
    "ImageParams",
    "Model",
    "Provider",
    "Stream",
    "StreamExecutor",
    "StreamFn",
    "StreamRequest",
    "VideoParams",
    # Provider factories
    "ai_gateway",
    "anthropic",
    "openai",
    # Adapter registration
    "register_generate",
    "register_stream",
    # Public API
    "check_connection",
    "generate",
    "stream",
]
