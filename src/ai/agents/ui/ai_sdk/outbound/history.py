"""Persisted-message → UIMessage list for history endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import _parts, _roundtrip, ui_message

if TYPE_CHECKING:
    from .....types import messages as messages_


def _message_turn_key(message: messages_.Message) -> str | None:
    return message.turn_id


def _assistant_bubble_id(message: messages_.Message) -> str:
    return _message_turn_key(message) or message.id


def _belongs_to_bubble(
    message: messages_.Message,
    bubble_id: str,
) -> bool:
    key = _message_turn_key(message)
    return key is None or key == bubble_id


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
                    metadata=_roundtrip.metadata_for([msg]),
                    parts=_parts.to_ui_parts(msg.parts),
                )
            )
            i += 1
            continue

        if msg.role == "assistant":
            ui_parts: list[ui_message.UIMessagePart] = []
            source_messages: list[messages_.Message] = []
            bubble_id = _assistant_bubble_id(msg)

            while i < len(messages) and messages[i].role in (
                "assistant",
                "tool",
                "internal",
            ):
                current = messages[i]
                if not _belongs_to_bubble(current, bubble_id):
                    break
                source_messages.append(current)
                if current.role == "assistant":
                    ui_parts.extend(_parts.to_ui_parts(current.parts))
                    ui_parts = _parts.dedupe_tool_parts(ui_parts)
                elif current.role == "tool":
                    _parts.merge_tool_results(ui_parts, current.parts)
                elif current.role == "internal":
                    _parts.merge_approval_signals(ui_parts, current.parts)
                i += 1
            ui_parts = _parts.dedupe_tool_parts(ui_parts)

            result.append(
                ui_message.UIMessage(
                    id=bubble_id,
                    role="assistant",
                    metadata=_roundtrip.metadata_for(source_messages),
                    parts=ui_parts,
                )
            )
            continue

        # Orphan tool / internal messages — skip; they have no assistant anchor.
        i += 1

    return result
