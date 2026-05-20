from __future__ import annotations

from typing import Any

from ai.agents.ui.ai_sdk import approvals
from ai.types import messages as messages_


def test_tool_call_id_for_strips_prefix() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="approve_tc_42",
        hook_type="ToolApproval",
        status="pending",
    )
    assert approvals.tool_call_id_for(hook) == "tc_42"


def test_tool_call_id_for_rejects_non_approval_type() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="approve_tc_42",
        hook_type="SomethingElse",
        status="pending",
    )
    assert approvals.tool_call_id_for(hook) is None


def test_tool_call_id_for_rejects_bad_prefix() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="tc_42",
        hook_type="ToolApproval",
        status="pending",
    )
    assert approvals.tool_call_id_for(hook) is None
