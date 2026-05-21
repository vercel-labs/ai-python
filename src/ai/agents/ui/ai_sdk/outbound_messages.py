"""Persisted-message conversion for AI SDK UI messages."""

from __future__ import annotations

from typing import Any, cast

from ....types import media
from ....types import messages as messages_
from . import approvals, id_utils, ui_messages
from .tool_utils import normalize_tool_input

UIToolLike = ui_messages.UIToolPart | ui_messages.UIDynamicToolPart

_TOOL_STATE_RANK: dict[ui_messages.UIToolInvocationState, int] = {
    "input-streaming": 0,
    "input-available": 1,
    "approval-requested": 2,
    "approval-responded": 3,
    "output-denied": 4,
    "output-error": 5,
    "output-available": 6,
}


def to_ui_parts(parts: list[messages_.Part]) -> list[ui_messages.UIMessagePart]:
    """Convert internal parts to UI message parts."""
    result: list[ui_messages.UIMessagePart] = []
    for part in parts:
        if isinstance(part, messages_.TextPart) and part.text:
            result.append(
                ui_messages.UITextPart.model_validate(
                    {
                        "type": "text",
                        "text": part.text,
                        "providerMetadata": part.provider_metadata,
                    }
                )
            )
        elif isinstance(part, messages_.ReasoningPart) and part.text:
            result.append(
                ui_messages.UIReasoningPart.model_validate(
                    {
                        "type": "reasoning",
                        "text": part.text,
                        "providerMetadata": part.provider_metadata,
                    }
                )
            )
        elif isinstance(part, messages_.ToolCallPart):
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
        elif isinstance(part, messages_.BuiltinToolCallPart):
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
        elif isinstance(part, messages_.BuiltinToolReturnPart):
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
        elif isinstance(part, messages_.FilePart):
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
    if candidate.output is not None:
        updates["output"] = candidate.output
    if candidate.raw_input is not None:
        updates["raw_input"] = candidate.raw_input
    if candidate.error_text is not None:
        updates["error_text"] = candidate.error_text
    if candidate.approval is not None:
        updates["approval"] = candidate.approval
    if candidate.provider_executed is not None:
        updates["provider_executed"] = candidate.provider_executed
    if candidate.call_provider_metadata is not None:
        updates["call_provider_metadata"] = candidate.call_provider_metadata
    if candidate.result_provider_metadata is not None:
        updates["result_provider_metadata"] = candidate.result_provider_metadata
    if candidate.tool_metadata is not None:
        updates["tool_metadata"] = candidate.tool_metadata
    if candidate.preliminary is not None:
        updates["preliminary"] = candidate.preliminary
    if candidate.title is not None:
        updates["title"] = candidate.title

    return existing.model_copy(update=updates) if updates else existing


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


def merge_tool_results(
    ui_parts: list[ui_messages.UIMessagePart],
    tool_parts: list[messages_.Part],
) -> None:
    """Merge tool result parts into existing UI tool parts."""
    tool_index: dict[str, int] = {}
    for idx, ui_part in enumerate(ui_parts):
        if isinstance(
            ui_part, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
        ):
            tool_index[ui_part.tool_call_id] = idx

    for part in tool_parts:
        if isinstance(part, messages_.ToolResultPart):
            tool_call_id = part.tool_call_id
            state = "output-error" if part.is_error else "output-available"
            updates: dict[str, Any] = {
                "state": state,
                "result_provider_metadata": part.provider_metadata,
            }
            if part.is_error:
                updates["error_text"] = str(part.result)
            else:
                updates["output"] = part.result
        elif isinstance(part, messages_.BuiltinToolReturnPart):
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
        else:
            continue

        if isinstance(part, messages_.ToolResultPart) and part.is_hook_pending:
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
    tool_index: dict[str, int] = {}
    for idx, ui_part in enumerate(ui_parts):
        if isinstance(
            ui_part, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
        ):
            tool_index[ui_part.tool_call_id] = idx

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
        if part.status == "pending":
            updates["state"] = "approval-requested"
            updates["approval"] = ui_messages.UIToolApproval.model_validate(
                {
                    "id": part.hook_id,
                    "isAutomatic": is_automatic,
                }
            )
        elif part.status == "resolved":
            resolution = cast(
                "dict[str, Any]",
                part.resolution if isinstance(part.resolution, dict) else {},
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
        elif part.status == "cancelled":
            updates["state"] = "output-error"
            updates["error_text"] = "Hook cancelled"

        if updates:
            ui_parts[idx] = existing.model_copy(update=updates)


def to_ui_messages(
    messages: list[messages_.Message],
) -> list[ui_messages.UIMessage]:
    """Group persisted messages into UI message bubbles."""
    result: list[ui_messages.UIMessage] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role in ("user", "system"):
            result.append(
                ui_messages.UIMessage(
                    id=msg.id,
                    role=msg.role,
                    metadata=id_utils.metadata_for([msg]),
                    parts=to_ui_parts(msg.parts),
                )
            )
            i += 1
            continue

        if msg.role == "assistant":
            ui_parts: list[ui_messages.UIMessagePart] = []
            source_messages: list[messages_.Message] = []
            bubble_id = msg.turn_id or msg.id

            while i < len(messages) and messages[i].role in (
                "assistant",
                "tool",
                "internal",
            ):
                current = messages[i]
                if current.turn_id is not None and current.turn_id != bubble_id:
                    break
                source_messages.append(current)
                if current.role == "assistant":
                    ui_parts.extend(to_ui_parts(current.parts))
                    ui_parts = dedupe_tool_parts(ui_parts)
                elif current.role == "tool":
                    merge_tool_results(ui_parts, current.parts)
                elif current.role == "internal":
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
            continue

        i += 1

    return result
