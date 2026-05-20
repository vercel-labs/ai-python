"""Serialize the UI message stream as Server-Sent Events."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pydantic

from .. import ui_events
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


def serialize_event(event: ui_events.UIMessageStreamEvent) -> str:
    """Serialize a stream event to JSON with camelCase keys."""
    d = dataclasses.asdict(event)
    if isinstance(event, ui_events.UIDataEvent):
        d["type"] = event.type
        del d["data_type"]
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict, default=_json_default)


def format_sse(event: ui_events.UIMessageStreamEvent) -> str:
    """Format a stream event as an SSE data line."""
    return f"data: {serialize_event(event)}\n\n"


def format_done_sse() -> str:
    """Format the AI SDK UI stream termination marker."""
    return "data: [DONE]\n\n"


async def to_sse(
    events: AsyncIterable[events_.AgentEvent],
) -> AsyncGenerator[str]:
    """Convert an internal event stream into SSE strings."""
    async for event in to_stream(events):
        yield format_sse(event)
    yield format_done_sse()
