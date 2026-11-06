"""Seeding Codex with a past: Responses API items for `thread/inject_items`.

The rule for what becomes what is the same for every CLI: see `_handoff`.
"""

from __future__ import annotations

from typing import Any

from ...types import media
from ...types import messages as messages_
from .. import _handoff

NOTE_PREFIX = "[Handoff from a previous session — earlier reasoning: "


def _note(reasoning: list[str]) -> str:
    return (
        NOTE_PREFIX + " ".join(r.strip() for r in reasoning if r.strip()) + "]"
    )


def _image_url(part: messages_.FilePart) -> str:
    """Return an http(s) URL as it is, anything else as a data URL."""
    return media.data_to_data_url(part.data, part.media_type)


def to_codex_items(messages: list[messages_.Message]) -> list[dict[str, Any]]:
    """Responses API items for `thread/inject_items`."""
    items: list[dict[str, Any]] = []
    for m in _handoff.harness_view(messages):
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
            elif isinstance(p, messages_.ToolCallPart):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": p.tool_call_id or p.id,
                        "name": p.tool_name,
                        "arguments": p.tool_args or "{}",
                    }
                )
            elif isinstance(p, messages_.ToolResultPart):
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
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _note(reasoning)}
                    ],
                }
            )
    return items
