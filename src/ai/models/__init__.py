"""models — composable model layer.

Usage::

    import ai
    model = ai.get_model("openai:gpt-5.4")
    provider = ai.get_provider("openai", base_url="http://localhost:11434/v1")
    model = ai.Model("llama3", provider=provider)
    model = ai.get_model("anthropic:claude-sonnet-4-6")
    provider = ai.get_provider("anthropic", base_url="https://anthropic.example.com")
    model = ai.Model("claude-sonnet-4-6", provider=provider)
    model = ai.get_model("anthropic/claude-sonnet-4")  # defaults to Gateway

    # stream — auto-creates client from env vars
    msgs = [ai.user_message("hello")]
    async with ai.stream(model, msgs) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)

    # explicit provider for custom auth / transport
    provider = ai.get_provider(
        "openai",
        base_url="https://custom.example.com/v1",
        api_key="sk-...",
    )
    model = ai.Model("gpt-5.4", provider=provider)
    async with ai.stream(model, msgs) as s:
        ...

    # list available models
    ids = await ai.get_provider("openai").list()
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
    generate,
    probe,
    stream,
)
from .core.model import Model, get_model
from .core.params import GenerateParams, ImageParams, VideoParams
from .core.proto import GenerateFn, StreamFn

__all__ = [
    # Core types
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
    "generate",
    "get_model",
    "probe",
    "stream",
]
