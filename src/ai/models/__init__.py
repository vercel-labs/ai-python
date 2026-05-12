"""models — composable model layer.

Usage::

    import ai
    from ai.providers import openai, openai_like, anthropic, anthropic_like, ai_gateway

    model = openai("gpt-5.4")
    model = openai_like(name="local", base_url="http://localhost:11434/v1")("llama3")
    model = anthropic("claude-sonnet-4-6")
    model = anthropic_like(name="custom", base_url="https://anthropic.example.com")(
        "claude-sonnet-4-6"
    )
    model = ai_gateway("anthropic/claude-sonnet-4")

    # stream — auto-creates client from env vars
    msgs = [ai.user_message("hello")]
    async with ai.stream(model, msgs) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)

    # explicit client for custom auth
    client = ai.Client(base_url="https://custom.example.com/v1", api_key="sk-...")
    model = openai("gpt-5.4", client=client)
    async with ai.stream(model, msgs) as s:
        ...

    # list available models
    ids = await openai.list()
"""

from ..providers.base import Provider
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
from .core.proto import CheckConnFn, GenerateFn, StreamFn

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
    # Adapter registration
    "register_generate",
    "register_stream",
    # Public API
    "check_connection",
    "generate",
    "stream",
]
