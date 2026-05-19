"""Approval-prefix linkage between ToolApproval hooks and tool calls.

TODO(datamodel-rework §4): once ``HookPart.target_tool_call_id`` is added,
delete this module and replace call sites with direct field access.
"""

from __future__ import annotations

from typing import Any

from ....types import messages as messages_
from ...hooks import TOOL_APPROVAL_HOOK_TYPE

_PREFIX = "approve_"


def tool_call_id_for(hook_part: messages_.HookPart[Any]) -> str | None:
    """Return the tool_call_id encoded in a ToolApproval hook id, or None."""
    if hook_part.hook_type != TOOL_APPROVAL_HOOK_TYPE:
        return None
    if hook_part.hook_id.startswith(_PREFIX):
        return hook_part.hook_id[len(_PREFIX) :]
    return None


def is_tool_approval_message(msg: messages_.Message) -> bool:
    """Return whether every part of ``msg`` is a ToolApproval HookPart."""
    if not msg.parts:
        return False
    return all(
        isinstance(p, messages_.HookPart) and tool_call_id_for(p) is not None
        for p in msg.parts
    )
