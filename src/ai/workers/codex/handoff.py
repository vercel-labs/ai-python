"""Seeding Codex with a past: Responses API items for `thread/inject_items`.

The rule for what becomes what is the same for every CLI: see `_handoff`.
"""

from __future__ import annotations

from typing import Any

from ...types import media
from ...types import messages as messages_
from .. import _handoff

NOTE_PREFIX = "[Handoff from a previous session — earlier reasoning: "


def _note(reasoning: list[str]) -> dict[str, Any]:
    """Return the reasoning of a turn as a user-role note."""
    text = " ".join(r.strip() for r in reasoning if r.strip())
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": NOTE_PREFIX + text + "]"}],
    }


def _image_url(part: messages_.FilePart) -> str:
    """Return an http(s) URL as it is, anything else as a data URL."""
    return media.data_to_data_url(part.data, part.media_type)


def to_codex_items(messages: list[messages_.Message]) -> list[dict[str, Any]]:
    """Responses API items for `thread/inject_items`.

    The items are a flat list, so each message goes in as it is: a call and
    its result are two items that follow each other. That holds for a tool
    an `ai.Agent` ran (a `ToolCallPart`, its result in a `tool` message) and
    for a tool the CLI ran (built-in parts, the result in the same message).
    A turn's reasoning goes in as a note after its calls, before their
    results.
    """
    items: list[dict[str, Any]] = []
    for m in messages:
        reasoning: list[str] = []
        for p in m.parts:
            if isinstance(p, messages_.TextPart) and p.text.strip():
                ctype = "input_text" if m.role == "user" else "output_text"
                items.append(
                    {
                        "type": "message",
                        "role": m.role,
                        "content": [{"type": ctype, "text": p.text}],
                    }
                )
            elif (
                isinstance(p, messages_.FilePart)
                and m.role == "user"
                and p.media_type.startswith("image/")
            ):
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": _image_url(p)}
                        ],
                    }
                )
            elif isinstance(
                p, messages_.ToolCallPart | messages_.BuiltinToolCallPart
            ):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": p.tool_call_id or p.id,
                        "name": p.tool_name,
                        "arguments": p.tool_args or "{}",
                    }
                )
            elif isinstance(
                p, messages_.ToolResultPart | messages_.BuiltinToolReturnPart
            ):
                if reasoning:
                    items.append(_note(reasoning))
                    reasoning = []
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": p.tool_call_id,
                        "output": _handoff.result_text(p.result),
                    }
                )
            elif isinstance(p, messages_.ReasoningPart) and p.text.strip():
                reasoning.append(p.text)
        if reasoning:
            items.append(_note(reasoning))
    return items
