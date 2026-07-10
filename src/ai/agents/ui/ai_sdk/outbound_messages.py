"""Persisted-message conversion for AI SDK UI messages."""

from __future__ import annotations

import json
from typing import Any, cast

from ....types import media
from ....types import messages as messages_
from . import approvals, id_utils, ui_messages
from .tool_utils import normalize_tool_input

UIToolLike = ui_messages.UIToolPart | ui_messages.UIDynamicToolPart

# Internal history can contain separate records for one tool call
# (call, approval, result). AI SDK UI expects one tool part per
# toolCallId, so later/higher-ranked states update the first part
# https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message#tooluipart
# https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol

_TOOL_STATE_RANK: dict[ui_messages.UIToolInvocationState, int] = {
    "input-streaming": 0,
    "input-available": 1,
    "approval-requested": 2,
    "approval-responded": 3,
    "output-denied": 4,
    "output-error": 5,
    "output-available": 6,
}

_MERGEABLE_TOOL_PART_FIELDS = (
    "raw_input",
    "output",
    "error_text",
    "approval",
    "provider_executed",
    "call_provider_metadata",
    "result_provider_metadata",
    "tool_metadata",
    "preliminary",
    "title",
)


def _merge_tool_part(
    existing: UIToolLike,
    candidate: UIToolLike,
) -> UIToolLike:
    """Merge duplicate UI tool parts, keeping the first display position."""
    existing_rank = _TOOL_STATE_RANK.get(existing.state, 0)
    candidate_rank = _TOOL_STATE_RANK.get(candidate.state, 0)
    updates: dict[str, Any] = {}

    if candidate_rank >= existing_rank:
        updates["state"] = candidate.state
        if candidate.state == "output-denied":
            updates["output"] = None

    if existing.input is None and candidate.input is not None:
        updates["input"] = candidate.input

    for field in _MERGEABLE_TOOL_PART_FIELDS:
        value = getattr(candidate, field)
        if value is not None:
            updates[field] = value

    return existing.model_copy(update=updates) if updates else existing


def _tool_part_index_by_call_id(
    ui_parts: list[ui_messages.UIMessagePart],
) -> dict[str, int]:
    return {
        ui_part.tool_call_id: idx
        for idx, ui_part in enumerate(ui_parts)
        if isinstance(
            ui_part, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
        )
    }


def dedupe_tool_parts(
    ui_parts: list[ui_messages.UIMessagePart],
) -> list[ui_messages.UIMessagePart]:
    """Collapse duplicate UI tool parts by tool_call_id."""
    result: list[ui_messages.UIMessagePart] = []
    tool_index: dict[str, int] = {}

    for part in ui_parts:
        if not isinstance(
            part, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
        ):
            result.append(part)
            continue

        idx = tool_index.get(part.tool_call_id)
        if idx is None:
            tool_index[part.tool_call_id] = len(result)
            result.append(part)
            continue

        existing = result[idx]
        if isinstance(
            existing, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
        ):
            result[idx] = _merge_tool_part(existing, part)

    return result


def bundle_to_wire_output(bundle: messages_.MessageBundle) -> Any:
    """Serialize a sub-agent transcript to its UI tool ``output``.

    Follows the AI SDK sub-agent convention of a single ``UIMessage`` for the
    common case (one bubble), and only falls back to a JSON list when the
    transcript spans multiple bubbles.  Returns ``None`` for an empty bundle
    so streaming callers can skip emitting until there's something to show.
    The inbound side accepts either shape (see ``_build_result_part``).
    """
    dumped = [
        m.model_dump(mode="json") for m in to_ui_messages(list(bundle.messages))
    ]
    if not dumped:
        return None
    return dumped[0] if len(dumped) == 1 else dumped


def _output_view(
    part: messages_.ToolResultPart,
) -> tuple[str, dict[str, Any]]:
    """Map a tool result to ``(state, field_updates)`` for the UI wire."""
    result = part.result
    if isinstance(result, messages_.ContentOutput):
        return "output-available", {
            "output": [item.model_dump(mode="json") for item in result.value]
        }
    if isinstance(result, messages_.MessageBundle):
        # `None` (empty bundle) becomes `[]` so a completed result still
        # round-trips to an (empty) MessageBundle rather than a null output.
        return "output-available", {
            "output": bundle_to_wire_output(result) or []
        }
    if part.is_error:
        text = result if isinstance(result, str) else json.dumps(result)
        return "output-error", {"error_text": text}
    return "output-available", {"output": result}


def merge_tool_results(
    ui_parts: list[ui_messages.UIMessagePart],
    tool_parts: list[messages_.Part],
) -> None:
    """Merge tool result parts into existing UI tool parts."""
    tool_index = _tool_part_index_by_call_id(ui_parts)

    for part in tool_parts:
        updates: dict[str, Any]
        match part:
            case messages_.ToolResultPart() if part.is_hook_deferred:
                continue
            case messages_.ToolResultPart():
                tool_call_id = part.tool_call_id
                state, field_updates = _output_view(part)
                updates = {
                    "state": state,
                    "result_provider_metadata": part.provider_metadata,
                    **field_updates,
                }
            case messages_.BuiltinToolReturnPart():
                tool_call_id = part.tool_call_id
                updates = {
                    "state": (
                        "output-error" if part.is_error else "output-available"
                    ),
                    "provider_executed": True,
                    "result_provider_metadata": part.provider_metadata,
                }
                if part.is_error:
                    updates["error_text"] = str(part.result)
                else:
                    updates["output"] = part.result
            case _:
                continue

        idx_opt = tool_index.get(tool_call_id)
        if idx_opt is None:
            continue
        idx = idx_opt
        existing = ui_parts[idx]
        if not isinstance(
            existing, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
        ):
            continue
        if existing.state == "output-denied":
            continue
        ui_parts[idx] = existing.model_copy(update=updates)


def merge_approval_signals(
    ui_parts: list[ui_messages.UIMessagePart],
    internal_parts: list[messages_.Part],
) -> None:
    """Merge approval hook state into existing UI tool parts."""
    tool_index = _tool_part_index_by_call_id(ui_parts)

    for part in internal_parts:
        if not isinstance(part, messages_.HookPart):
            continue

        tool_call_id = approvals.tool_call_id_for(part)
        if tool_call_id is None:
            continue

        idx_opt = tool_index.get(tool_call_id)
        if idx_opt is None:
            continue
        idx = idx_opt

        existing = ui_parts[idx]
        if not isinstance(
            existing, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
        ):
            continue

        updates: dict[str, Any] = {}
        provider_executed = part.metadata.get("providerExecuted")
        if isinstance(provider_executed, bool):
            updates["provider_executed"] = provider_executed
        is_automatic = part.metadata.get("isAutomatic")
        is_automatic = is_automatic if isinstance(is_automatic, bool) else None
        match part.status:
            case "pending":
                updates["state"] = "approval-requested"
                updates["approval"] = ui_messages.UIToolApproval.model_validate(
                    {
                        "id": part.hook_id,
                        "isAutomatic": is_automatic,
                    }
                )
            case "resolved":
                resolution = cast(
                    "dict[str, Any]",
                    part.resolution
                    if isinstance(part.resolution, dict)
                    else {},
                )
                updates["approval"] = ui_messages.UIToolApproval.model_validate(
                    {
                        "id": part.hook_id,
                        "approved": resolution.get("granted"),
                        "reason": resolution.get("reason"),
                        "isAutomatic": is_automatic,
                    }
                )
                if resolution.get("granted", False):
                    updates["state"] = "approval-responded"
                else:
                    updates["state"] = "output-denied"
                    updates["output"] = None
            case "cancelled":
                updates["state"] = "output-error"
                updates["error_text"] = "Hook cancelled"

        if updates:
            ui_parts[idx] = existing.model_copy(update=updates)


def to_ui_parts(parts: list[messages_.Part]) -> list[ui_messages.UIMessagePart]:
    """Convert internal parts to UI message parts."""
    result: list[ui_messages.UIMessagePart] = []
    for part in parts:
        match part:
            case messages_.TextPart(text=text) if text:
                result.append(
                    ui_messages.UITextPart.model_validate(
                        {
                            "type": "text",
                            "text": text,
                            "providerMetadata": part.provider_metadata,
                        }
                    )
                )
            case messages_.ReasoningPart(text=text) if text:
                result.append(
                    ui_messages.UIReasoningPart.model_validate(
                        {
                            "type": "reasoning",
                            "text": text,
                            "providerMetadata": part.provider_metadata,
                        }
                    )
                )
            case messages_.ToolCallPart():
                result.append(
                    ui_messages.UIToolPart.model_validate(
                        {
                            "type": f"tool-{part.tool_name}",
                            "toolCallId": part.tool_call_id,
                            "state": "input-available",
                            "input": normalize_tool_input(part.tool_args),
                            "callProviderMetadata": part.provider_metadata,
                        }
                    )
                )
            case messages_.BuiltinToolCallPart():
                result.append(
                    ui_messages.UIDynamicToolPart.model_validate(
                        {
                            "type": "dynamic-tool",
                            "toolName": part.tool_name,
                            "toolCallId": part.tool_call_id,
                            "state": "input-available",
                            "input": normalize_tool_input(part.tool_args),
                            "providerExecuted": True,
                            "callProviderMetadata": part.provider_metadata,
                        }
                    )
                )
            case messages_.BuiltinToolReturnPart():
                result.append(
                    ui_messages.UIDynamicToolPart.model_validate(
                        {
                            "type": "dynamic-tool",
                            "toolName": part.tool_name,
                            "toolCallId": part.tool_call_id,
                            "state": (
                                "output-error"
                                if part.is_error
                                else "output-available"
                            ),
                            "input": None,
                            "output": None if part.is_error else part.result,
                            "errorText": (
                                str(part.result) if part.is_error else None
                            ),
                            "providerExecuted": True,
                            "resultProviderMetadata": part.provider_metadata,
                        }
                    )
                )
            case messages_.FilePart():
                result.append(
                    ui_messages.UIFilePart.model_validate(
                        {
                            "type": "file",
                            "mediaType": part.media_type,
                            "url": media.data_to_data_url(
                                part.data, part.media_type
                            ),
                            "filename": part.filename,
                            "providerMetadata": part.provider_metadata,
                        }
                    )
                )
    return result


def to_ui_messages(
    messages: list[messages_.Message],
) -> list[ui_messages.UIMessage]:
    """Group persisted messages into UI message bubbles."""
    result: list[ui_messages.UIMessage] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        match msg.role:
            case "user" | "system":
                result.append(
                    ui_messages.UIMessage(
                        id=msg.id,
                        role=msg.role,
                        metadata=id_utils.metadata_for([msg]),
                        parts=to_ui_parts(msg.parts),
                    )
                )
                i += 1
            case "assistant":
                ui_parts: list[ui_messages.UIMessagePart] = []
                source_messages: list[messages_.Message] = []
                bubble_id = msg.turn_id or msg.id

                while i < len(messages) and messages[i].role in (
                    "assistant",
                    "tool",
                    "internal",
                ):
                    current = messages[i]
                    if (
                        current.turn_id is not None
                        and current.turn_id != bubble_id
                    ):
                        break
                    source_messages.append(current)
                    match current.role:
                        case "assistant":
                            ui_parts.extend(to_ui_parts(current.parts))
                            ui_parts = dedupe_tool_parts(ui_parts)
                        case "tool":
                            merge_tool_results(ui_parts, current.parts)
                        case "internal":
                            merge_approval_signals(ui_parts, current.parts)
                    i += 1
                ui_parts = dedupe_tool_parts(ui_parts)

                result.append(
                    ui_messages.UIMessage(
                        id=bubble_id,
                        role="assistant",
                        metadata=id_utils.metadata_for(source_messages),
                        parts=ui_parts,
                    )
                )
            case _:
                i += 1

    return result
