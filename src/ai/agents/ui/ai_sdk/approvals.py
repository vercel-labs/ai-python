"""Tool approval helpers for AI SDK UI adapters."""

from __future__ import annotations

from typing import Any, NamedTuple

from ....types import messages as messages_
from ...hooks import TOOL_APPROVAL_HOOK_TYPE, resolve_hook
from . import ui_messages

_PREFIX = "approve_"


ToolPart = ui_messages.UIToolPart | ui_messages.UIDynamicToolPart


class ApprovalResponse(NamedTuple):
    """Approval response extracted from a responded UI tool part."""

    hook_id: str
    granted: bool
    reason: str | None
    tool_call_id: str


def tool_call_id_for(hook_part: messages_.HookPart[Any]) -> str | None:
    """Return the tool_call_id encoded in a ToolApproval hook id, or None."""
    if hook_part.hook_type != TOOL_APPROVAL_HOOK_TYPE:
        return None
    if hook_part.hook_id.startswith(_PREFIX):
        return hook_part.hook_id[len(_PREFIX) :]
    return None


def metadata_bool(metadata: dict[str, Any], key: str) -> bool | None:
    value = metadata.get(key)
    return value if isinstance(value, bool) else None


def metadata_dict(
    metadata: dict[str, Any],
    key: str,
) -> dict[str, Any] | None:
    value = metadata.get(key)
    return value if isinstance(value, dict) else None


def metadata_from_tool_part(tp: ToolPart) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if tp.approval is not None and tp.approval.is_automatic is not None:
        metadata["isAutomatic"] = tp.approval.is_automatic
    if tp.provider_executed is not None:
        metadata["providerExecuted"] = tp.provider_executed
    if tp.call_provider_metadata is not None:
        metadata["callProviderMetadata"] = tp.call_provider_metadata
    return metadata


def hook_part_from_tool_part(tp: ToolPart) -> messages_.HookPart[Any] | None:
    """Reconstruct approval hook state from a UI tool part when possible."""
    approval = tp.approval
    if approval is None:
        return None

    metadata = metadata_from_tool_part(tp)

    if tp.state == "approval-requested":
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type=TOOL_APPROVAL_HOOK_TYPE,
            status="pending",
            metadata=metadata,
        )

    if tp.state == "approval-responded" and approval.approved is not None:
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type=TOOL_APPROVAL_HOOK_TYPE,
            status="resolved",
            metadata=metadata,
            resolution={
                "granted": approval.approved,
                "reason": approval.reason,
            },
        )

    if tp.state == "output-denied":
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type=TOOL_APPROVAL_HOOK_TYPE,
            status="resolved",
            metadata=metadata,
            resolution={
                "granted": False,
                "reason": approval.reason,
            },
        )

    return None


def extract_approvals(
    ui_messages_list: list[ui_messages.UIMessage],
) -> list[ApprovalResponse]:
    """Return every approval response found in UI messages."""
    approvals: list[ApprovalResponse] = []
    for ui_msg in ui_messages_list:
        for part in ui_msg.parts:
            if not isinstance(
                part, ui_messages.UIToolPart | ui_messages.UIDynamicToolPart
            ):
                continue
            if (
                part.state == "approval-responded"
                and part.approval is not None
                and part.approval.approved is not None
            ):
                approvals.append(
                    ApprovalResponse(
                        hook_id=part.approval.id,
                        granted=part.approval.approved,
                        reason=part.approval.reason,
                        tool_call_id=part.tool_call_id,
                    )
                )
    return approvals


def apply_approvals(approvals: list[ApprovalResponse]) -> None:
    """Pre-register each approval resolution with the hooks registry."""
    for approval in approvals:
        resolve_hook(
            approval.hook_id,
            {"granted": approval.granted, "reason": approval.reason},
        )


def is_resolved_approval_message(msg: messages_.Message) -> bool:
    """Return whether ``msg`` records a resolved tool approval hook."""
    if msg.role != "internal" or len(msg.parts) != 1:
        return False
    part = msg.parts[0]
    return (
        isinstance(part, messages_.HookPart)
        and part.hook_type == TOOL_APPROVAL_HOOK_TYPE
        and part.status == "resolved"
    )
