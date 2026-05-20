"""Roundtrip metadata for preserving internal message identity."""

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


def source_message_entry(message: messages_.Message) -> dict[str, object]:
    return {
        "id": message.id,
        "role": message.role,
        "turnId": message.turn_id,
        "partIds": [part.id for part in message.parts],
    }


def metadata_for(
    source_messages: list[messages_.Message],
) -> dict[str, object]:
    return {
        ADAPTER_METADATA_KEY: {
            SOURCE_MESSAGES_KEY: [
                source_message_entry(message) for message in source_messages
            ]
        }
    }


def source_messages_from(metadata: object) -> list[SourceMessage]:
    if not isinstance(metadata, dict):
        return []

    adapter_metadata = metadata.get(ADAPTER_METADATA_KEY)
    if not isinstance(adapter_metadata, dict):
        return []

    raw_source_messages = adapter_metadata.get(SOURCE_MESSAGES_KEY)
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
    if not source_messages:
        return messages

    restored: list[messages_.Message] = []
    source_index = 0

    for message in messages:
        match_index = _find_next_source(
            source_messages,
            role=message.role,
            start=source_index,
        )
        if match_index is None:
            restored.append(message)
            continue

        source = source_messages[match_index]
        source_index = match_index + 1
        restored.append(_restore_message_ids(message, source))

    return restored


def _parse_source_message(raw: object) -> SourceMessage | None:
    if not isinstance(raw, dict):
        return None

    message_id = raw.get("id")
    role = raw.get("role")
    if not isinstance(message_id, str) or role not in _VALID_ROLES:
        return None

    raw_turn_id = raw.get("turnId")
    turn_id = raw_turn_id if isinstance(raw_turn_id, str) else None

    raw_part_ids = raw.get("partIds")
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


def _find_next_source(
    source_messages: list[SourceMessage],
    *,
    role: MessageRole,
    start: int,
) -> int | None:
    for index in range(start, len(source_messages)):
        if source_messages[index].role == role:
            return index
    return None


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
