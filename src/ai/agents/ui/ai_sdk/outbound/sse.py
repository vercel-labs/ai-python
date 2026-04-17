"""Serialize the UI message stream as Server-Sent Events."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncGenerator, AsyncIterable

from .....types import messages as messages_
from .. import protocol
from .stream import to_stream


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def serialize_part(part: protocol.UIMessageStreamPart) -> str:
    """Serialize a stream part to JSON with camelCase keys."""
    d = dataclasses.asdict(part)
    if isinstance(part, protocol.DataPart):
        d["type"] = part.type
        del d["data_type"]
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict)


def format_sse(part: protocol.UIMessageStreamPart) -> str:
    """Format a stream part as an SSE data line."""
    return f"data: {serialize_part(part)}\n\n"


async def to_sse(
    messages: AsyncIterable[messages_.Message],
) -> AsyncGenerator[str]:
    """Convert an internal message stream into SSE strings."""
    async for part in to_stream(messages):
        yield format_sse(part)
