"""Persisted-message → UIMessage list for history endpoints."""

from __future__ import annotations

from .....types import messages as messages_
from .. import _parts, ui_message


def to_ui_messages(
    messages: list[messages_.Message],
) -> list[ui_message.UIMessage]:
    """Group persisted messages into UIMessage bubbles.

    ``user``/``system`` messages become standalone UIMessages.  Runs of
    ``assistant``/``tool``/``internal`` messages merge into a single
    assistant UIMessage, with tool results and approval state folded into
    the corresponding tool-call parts.
    """
    result: list[ui_message.UIMessage] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role in ("user", "system"):
            result.append(
                ui_message.UIMessage(
                    id=msg.id,
                    role=msg.role,
                    parts=_parts.to_ui_parts(msg.parts),
                )
            )
            i += 1
            continue

        if msg.role == "assistant":
            ui_parts: list[ui_message.UIMessagePart] = []
            bubble_id = msg.id

            while i < len(messages) and messages[i].role in (
                "assistant",
                "tool",
                "internal",
            ):
                current = messages[i]
                if current.role == "assistant":
                    ui_parts.extend(_parts.to_ui_parts(current.parts))
                elif current.role == "tool":
                    _parts.merge_tool_results(ui_parts, current.parts)
                elif current.role == "internal":
                    _parts.merge_approval_signals(ui_parts, current.parts)
                i += 1

            result.append(
                ui_message.UIMessage(
                    id=bubble_id,
                    role="assistant",
                    parts=ui_parts,
                )
            )
            continue

        # Orphan tool / internal messages — skip; they have no assistant anchor.
        i += 1

    return result
