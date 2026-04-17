from __future__ import annotations

from ai.agents.ui.ai_sdk import _approvals
from ai.types import messages as messages_


def test_tool_call_id_for_strips_prefix() -> None:
    hook = messages_.HookPart(
        hook_id="approve_tc_42",
        hook_type="ToolApproval",
        status="pending",
    )
    assert _approvals.tool_call_id_for(hook) == "tc_42"


def test_tool_call_id_for_rejects_non_approval_type() -> None:
    hook = messages_.HookPart(
        hook_id="approve_tc_42",
        hook_type="SomethingElse",
        status="pending",
    )
    assert _approvals.tool_call_id_for(hook) is None


def test_tool_call_id_for_rejects_bad_prefix() -> None:
    hook = messages_.HookPart(
        hook_id="tc_42",
        hook_type="ToolApproval",
        status="pending",
    )
    assert _approvals.tool_call_id_for(hook) is None


def test_is_tool_approval_message_detects_all_approval_hooks() -> None:
    msg = messages_.Message(
        role="internal",
        parts=[
            messages_.HookPart(
                hook_id="approve_tc_1",
                hook_type="ToolApproval",
                status="pending",
            ),
        ],
    )
    assert _approvals.is_tool_approval_message(msg)


def test_is_tool_approval_message_false_for_non_approval() -> None:
    msg = messages_.Message(
        role="internal",
        parts=[
            messages_.HookPart(
                hook_id="other",
                hook_type="Something",
                status="pending",
            ),
        ],
    )
    assert not _approvals.is_tool_approval_message(msg)


def test_is_tool_approval_message_false_for_empty() -> None:
    msg = messages_.Message(role="internal", parts=[])
    assert not _approvals.is_tool_approval_message(msg)
