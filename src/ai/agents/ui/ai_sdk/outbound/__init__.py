"""Outbound adapter: ``ai.Message`` stream → AI SDK UI protocol."""

from .history import to_ui_messages
from .sse import to_sse
from .stream import to_stream

__all__ = ["to_stream", "to_sse", "to_ui_messages"]
