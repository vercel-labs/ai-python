"""Core types for models."""

from .adapters import register_check, register_generate, register_stream
from .api import check_connection, generate, stream
from .catalog import get_models, get_providers, register_catalog
from .catalog import model as model_factory
from .client import Client
from .model import Model, ModelCost
from .proto import CheckConnFn, GenerateFn, StreamFn
from .stream_result import StreamResult

__all__ = [
    "CheckConnFn",
    "Client",
    "GenerateFn",
    "Model",
    "ModelCost",
    "StreamFn",
    "StreamResult",
    "check_connection",
    "generate",
    "get_models",
    "get_providers",
    "model_factory",
    "register_catalog",
    "register_check",
    "register_generate",
    "register_stream",
    "stream",
]
