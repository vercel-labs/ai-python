"""Core types for models."""

from .adapters import register_generate, register_stream
from .api import (
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
from .client import Client
from .model import Model
from .params import GenerateParams, ImageParams, VideoParams
from .proto import CheckConnFn, GenerateFn, Provider, StreamFn

__all__ = [
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
    "check_connection",
    "generate",
    "register_generate",
    "register_stream",
    "stream",
]
