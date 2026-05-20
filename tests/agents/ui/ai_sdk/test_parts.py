from __future__ import annotations

from ai.agents.ui.ai_sdk import _parts
from ai.agents.ui.ai_sdk.ui_messages import (
    UIReasoningPart,
    UITextPart,
    UIToolApproval,
    UIToolPart,
)
from ai.types import messages as messages_


def test_to_ui_parts_text_and_reasoning() -> None:
    parts: list[messages_.Part] = [
        messages_.ReasoningPart(text="thinking"),
        messages_.TextPart(text="hi"),
    ]
    ui_parts = _parts.to_ui_parts(parts)
    assert isinstance(ui_parts[0], UIReasoningPart)
    assert ui_parts[0].text == "thinking"
    assert isinstance(ui_parts[1], UITextPart)
    assert ui_parts[1].text == "hi"


def test_to_ui_parts_tool_call_parses_json_args() -> None:
    parts: list[messages_.Part] = [
        messages_.ToolCallPart(
            tool_call_id="tc1",
            tool_name="search",
            tool_args='{"q": "x"}',
        )
    ]
    ui_parts = _parts.to_ui_parts(parts)
    assert isinstance(ui_parts[0], UIToolPart)
    assert ui_parts[0].type == "tool-search"
    assert ui_parts[0].input == {"q": "x"}
    assert ui_parts[0].state == "input-available"


def test_merge_tool_results_updates_state_and_output() -> None:
    parts: list[messages_.Part] = [
        messages_.ToolCallPart(
            tool_call_id="tc1",
            tool_name="search",
            tool_args="{}",
        )
    ]
    ui_parts = _parts.to_ui_parts(parts)
    _parts.merge_tool_results(
        ui_parts,
        [
            messages_.ToolResultPart(
                tool_call_id="tc1",
                tool_name="search",
                result={"hits": 3},
            )
        ],
    )
    merged = ui_parts[0]
    assert isinstance(merged, UIToolPart)
    assert merged.state == "output-available"
    assert merged.output == {"hits": 3}


def test_merge_approval_signals_pending_then_resolved() -> None:
    parts: list[messages_.Part] = [
        messages_.ToolCallPart(
            tool_call_id="tc1",
            tool_name="delete",
            tool_args="{}",
        )
    ]
    ui_parts = _parts.to_ui_parts(parts)

    _parts.merge_approval_signals(
        ui_parts,
        [
            messages_.HookPart(
                hook_id="approve_tc1",
                hook_type="ToolApproval",
                status="pending",
            )
        ],
    )
    requested = ui_parts[0]
    assert isinstance(requested, UIToolPart)
    assert requested.state == "approval-requested"
    assert isinstance(requested.approval, UIToolApproval)

    _parts.merge_approval_signals(
        ui_parts,
        [
            messages_.HookPart(
                hook_id="approve_tc1",
                hook_type="ToolApproval",
                status="resolved",
                resolution={"granted": True, "reason": None},
            )
        ],
    )
    responded = ui_parts[0]
    assert isinstance(responded, UIToolPart)
    assert responded.state == "approval-responded"
    assert responded.approval is not None
    assert responded.approval.approved is True
