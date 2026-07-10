"""Tool approval helpers for AI SDK UI adapters."""

from __future__ import annotations

from typing import Any, NamedTuple

from ....types import messages as messages_
from ...hooks import TOOL_APPROVAL_HOOK_TYPE, HookRegistry, resolve_hook
from . import ui_messages

ToolPart = ui_messages.UIToolPart | ui_messages.UIDynamicToolPart


class ApprovalResponse(NamedTuple):
    """Approval response extracted from a responded UI tool part."""

    hook_id: str
    granted: bool
    reason: str | None
    tool_call_id: str


def tool_call_id_for(hook_part: messages_.HookPart[Any]) -> str | None:
    """Return the tool call a ToolApproval hook suspends, or None."""
    if hook_part.hook_type != TOOL_APPROVAL_HOOK_TYPE:
        return None
    return hook_part.tool_call_id


def hook_part_from_tool_part(tp: ToolPart) -> messages_.HookPart[Any] | None:
    """Reconstruct approval hook state from a UI tool part when possible."""
    approval = tp.approval
    if approval is None:
        return None

    metadata: dict[str, Any] = {}
    if approval.is_automatic is not None:
        metadata["isAutomatic"] = approval.is_automatic
    if tp.provider_executed is not None:
        metadata["providerExecuted"] = tp.provider_executed
    if tp.call_provider_metadata is not None:
        metadata["callProviderMetadata"] = tp.call_provider_metadata

    if tp.state == "approval-requested":
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type=TOOL_APPROVAL_HOOK_TYPE,
            status="pending",
            metadata=metadata,
            tool_call_id=tp.tool_call_id,
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
            tool_call_id=tp.tool_call_id,
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
            tool_call_id=tp.tool_call_id,
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


def apply_approvals(
    approvals: list[ApprovalResponse],
    *,
    registry: HookRegistry | None = None,
) -> None:
    """Pre-register each approval resolution with a hook registry.

    ``registry`` defaults to the current one, so either call this
    inside the ``agent.run()`` block (before iterating the stream), or
    pass the registry the run will use.
    """
    for approval in approvals:
        resolve_hook(
            approval.hook_id,
            {"granted": approval.granted, "reason": approval.reason},
            registry=registry,
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
