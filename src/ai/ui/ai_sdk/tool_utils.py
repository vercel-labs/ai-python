"""Tool value helpers shared by AI SDK UI adapters."""

from __future__ import annotations

import json
from typing import Any


def normalize_tool_input(raw: str) -> Any:
    """Parse serialized tool args into a JSON value when possible."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def normalize_tool_args(tool_input: Any) -> str:
    """Normalize UI tool input to the internal serialized args form."""
    match tool_input:
        case str():
            return tool_input
        case None:
            return "{}"
        case _:
            return json.dumps(tool_input)
