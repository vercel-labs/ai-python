"""Roundtrip metadata for preserving internal message identity.

The adapter writes ``metadata["aiPython"]["sourceMessages"]`` with each
source message's ``id``, ``role``, ``turnId``, and ``partIds``. Outbound UI
bubbles can collapse assistant/tool/internal messages into one UI message;
inbound parsing uses this metadata to restore stable message and part ids.

It also writes ``metadata["aiPython"]["toolResultKinds"]`` mapping a tool
call id to its ``result_kind`` for results the wire ``state`` can't convey
(``content``): the multipart payload already round-trips inside the UI tool
part's ``output``, but the signal to rehydrate it as a typed ``ContentOutput``
would otherwise be lost.  ``error`` rides the UI ``state`` enum and ``json``
is the default, so only ``content`` is recorded.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, cast

from ...types import messages as messages_

ADAPTER_METADATA_KEY = "aiPython"
SOURCE_MESSAGES_KEY = "sourceMessages"
TOOL_RESULT_KINDS_KEY = "toolResultKinds"

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


def _tool_result_kinds(
    source_messages: list[messages_.Message],
) -> dict[str, str]:
    """Collect ``{tool_call_id: subtype}`` for special tool results.

    The recorded value is the :class:`SpecialToolResult` discriminator so the
    inbound side can rehydrate the typed result without shape-sniffing it.
    """
    kinds: dict[str, str] = {}
    for message in source_messages:
        for part in message.parts:
            if isinstance(part, messages_.ToolResultPart) and isinstance(
                part.result, messages_.SpecialToolResult
            ):
                kinds[part.tool_call_id] = part.result.type
    return kinds


def metadata_for(
    source_messages: list[messages_.Message],
) -> dict[str, object]:
    """Return adapter metadata for restoring collapsed source message ids."""
    adapter: dict[str, object] = {
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
    tool_result_kinds = _tool_result_kinds(source_messages)
    if tool_result_kinds:
        adapter[TOOL_RESULT_KINDS_KEY] = tool_result_kinds
    return {ADAPTER_METADATA_KEY: adapter}


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


def tool_result_kinds_from(metadata: object) -> dict[str, str]:
    """Parse ``{tool_call_id: result_kind}``, ignoring malformed entries."""
    if not isinstance(metadata, dict):
        return {}
    metadata_dict = cast("dict[str, object]", metadata)
    adapter_metadata = metadata_dict.get(ADAPTER_METADATA_KEY)
    if not isinstance(adapter_metadata, dict):
        return {}
    adapter_metadata_dict = cast("dict[str, object]", adapter_metadata)
    raw_kinds = adapter_metadata_dict.get(TOOL_RESULT_KINDS_KEY)
    if not isinstance(raw_kinds, dict):
        return {}
    raw_kinds_dict = cast("dict[str, object]", raw_kinds)
    return {
        tool_call_id: kind
        for tool_call_id, kind in raw_kinds_dict.items()
        if isinstance(tool_call_id, str) and isinstance(kind, str)
    }


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
