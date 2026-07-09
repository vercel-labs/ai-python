from __future__ import annotations

from typing import Any

from ai.agents.ui.ai_sdk import approvals
from ai.agents.ui.ai_sdk.ui_messages import UIMessage
from ai.types import messages as messages_


def _ui(role: str, *parts: dict[str, Any], id: str = "m1") -> UIMessage:
    return UIMessage.model_validate(
        {"id": id, "role": role, "parts": list(parts)}
    )


def _tool(
    tool_name: str,
    tool_call_id: str,
    state: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": f"tool-{tool_name}",
        "toolCallId": tool_call_id,
        "state": state,
        **extra,
    }


def test_tool_call_id_for_returns_field() -> None:
    # Custom gating can use any label as long as tool_call_id is set.
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="my_custom_gate",
        hook_type="ToolApproval",
        status="deferred",
        tool_call_id="tc_42",
    )
    assert approvals.tool_call_id_for(hook) == "tc_42"


def test_tool_call_id_for_field_ignored_on_non_approval() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="confirm_something",
        hook_type="Confirmation",
        status="deferred",
        tool_call_id="tc_42",
    )
    assert approvals.tool_call_id_for(hook) is None


def test_tool_call_id_for_rejects_non_approval_type() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="approve_tc_42",
        hook_type="SomethingElse",
        status="deferred",
        tool_call_id="tc_42",
    )
    assert approvals.tool_call_id_for(hook) is None


def test_extract_approvals_returns_approved_responses() -> None:
    approval_responses = approvals.extract_approvals(
        [
            _ui(
                "assistant",
                _tool(
                    "x",
                    "tc1",
                    "approval-responded",
                    approval={
                        "id": "approve_tc1",
                        "approved": False,
                        "reason": "nope",
                    },
                ),
            )
        ]
    )
    assert len(approval_responses) == 1
    assert approval_responses[0].hook_id == "approve_tc1"
    assert approval_responses[0].granted is False
    assert approval_responses[0].reason == "nope"


def test_extract_approvals_handles_dynamic_tool_responses() -> None:
    approval_responses = approvals.extract_approvals(
        [
            _ui(
                "assistant",
                {
                    "type": "dynamic-tool",
                    "toolName": "web_search",
                    "toolCallId": "tc1",
                    "state": "approval-responded",
                    "input": {"query": "ai"},
                    "approval": {
                        "id": "approve_tc1",
                        "approved": True,
                        "reason": "ok",
                        "isAutomatic": True,
                    },
                    "providerExecuted": True,
                },
            )
        ]
    )

    assert len(approval_responses) == 1
    assert approval_responses[0].hook_id == "approve_tc1"
    assert approval_responses[0].granted is True
    assert approval_responses[0].reason == "ok"
    assert approval_responses[0].tool_call_id == "tc1"
