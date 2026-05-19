"""Serialize the UI message stream as Server-Sent Events."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pydantic

from .. import protocol
from .stream import to_stream

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable

    from .....types import events as events_


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _json_default(obj: Any) -> Any:
    """Fallback encoder for json.dumps — handle pydantic models recursively.

    Aggregator snapshots and tool outputs may carry pydantic models
    (e.g. ``MessageBundle``, ``UIMessage``).  ``model_dump(mode="json")``
    converts them to plain JSON-native dicts/lists.
    """
    if isinstance(obj, pydantic.BaseModel):
        return obj.model_dump(mode="json", by_alias=True)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def serialize_part(part: protocol.UIMessageStreamPart) -> str:
    """Serialize a stream part to JSON with camelCase keys."""
    d = dataclasses.asdict(part)
    if isinstance(part, protocol.DataPart):
        d["type"] = part.type
        del d["data_type"]
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict, default=_json_default)


def format_sse(part: protocol.UIMessageStreamPart) -> str:
    """Format a stream part as an SSE data line."""
    return f"data: {serialize_part(part)}\n\n"


async def to_sse(
    events: AsyncIterable[events_.AgentEvent],
) -> AsyncGenerator[str]:
    """Convert an internal event stream into SSE strings."""
    async for part in to_stream(events):
        yield format_sse(part)
