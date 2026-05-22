"""Roundtrip metadata for preserving internal message identity.

The adapter writes ``metadata["aiPython"]["sourceMessages"]`` with each
source message's ``id``, ``role``, ``turnId``, and ``partIds``. Outbound UI
bubbles can collapse assistant/tool/internal messages into one UI message;
inbound parsing uses this metadata to restore stable message and part ids.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from ....types import messages as messages_

ADAPTER_METADATA_KEY = "aiPython"
SOURCE_MESSAGES_KEY = "sourceMessages"

MessageRole = Literal["user", "assistant", "system", "tool", "internal"]
_VALID_ROLES = {"user", "assistant", "system", "tool", "internal"}


@dataclasses.dataclass(frozen=True)
class SourceMessage:
    id: str
    role: MessageRole
    turn_id: str | None
    part_ids: tuple[str, ...]


def _parse_source_message(raw: object) -> SourceMessage | None:
    if not isinstance(raw, dict):
        return None

    raw_dict = cast("dict[str, object]", raw)
    message_id = raw_dict.get("id")
    role = raw_dict.get("role")
    if not isinstance(message_id, str) or role not in _VALID_ROLES:
        return None

    raw_turn_id = raw_dict.get("turnId")
    turn_id = raw_turn_id if isinstance(raw_turn_id, str) else None

    raw_part_ids = raw_dict.get("partIds")
    part_ids = (
        tuple(part_id for part_id in raw_part_ids if isinstance(part_id, str))
        if isinstance(raw_part_ids, list)
        else ()
    )

    return SourceMessage(
        id=message_id,
        role=cast("MessageRole", role),
        turn_id=turn_id,
        part_ids=part_ids,
    )


def _restore_message_ids(
    message: messages_.Message,
    source: SourceMessage,
) -> messages_.Message:
    updates: dict[str, object] = {
        "id": source.id,
        "turn_id": source.turn_id,
    }

    if len(source.part_ids) == len(message.parts):
        updates["parts"] = [
            part.model_copy(update={"id": part_id})
            for part, part_id in zip(
                message.parts, source.part_ids, strict=True
            )
        ]

    return message.model_copy(update=updates)


def metadata_for(
    source_messages: list[messages_.Message],
) -> dict[str, object]:
    """Return adapter metadata for restoring collapsed source message ids."""
    return {
        ADAPTER_METADATA_KEY: {
            SOURCE_MESSAGES_KEY: [
                {
                    "id": message.id,
                    "role": message.role,
                    "turnId": message.turn_id,
                    "partIds": [part.id for part in message.parts],
                }
                for message in source_messages
            ]
        }
    }


def source_messages_from(metadata: object) -> list[SourceMessage]:
    """Parse adapter metadata, ignoring missing or malformed entries."""
    if not isinstance(metadata, dict):
        return []

    metadata_dict = cast("dict[str, object]", metadata)
    adapter_metadata = metadata_dict.get(ADAPTER_METADATA_KEY)
    if not isinstance(adapter_metadata, dict):
        return []

    adapter_metadata_dict = cast("dict[str, object]", adapter_metadata)
    raw_source_messages = adapter_metadata_dict.get(SOURCE_MESSAGES_KEY)
    if not isinstance(raw_source_messages, list):
        return []

    result: list[SourceMessage] = []
    for raw in raw_source_messages:
        source = _parse_source_message(raw)
        if source is not None:
            result.append(source)
    return result


def restore_source_ids(
    messages: list[messages_.Message],
    source_messages: list[SourceMessage],
) -> list[messages_.Message]:
    """Restore message and part ids from matching source metadata."""
    if not source_messages:
        return messages

    restored: list[messages_.Message] = []
    source_index = 0

    for message in messages:
        match_index = next(
            (
                index
                for index in range(source_index, len(source_messages))
                if source_messages[index].role == message.role
            ),
            None,
        )
        if match_index is None:
            restored.append(message)
            continue

        source = source_messages[match_index]
        source_index = match_index + 1
        restored.append(_restore_message_ids(message, source))

    return restored
