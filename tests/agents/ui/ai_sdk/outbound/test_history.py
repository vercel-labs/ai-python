from __future__ import annotations

from ai.agents.ui.ai_sdk import to_ui_messages
from ai.agents.ui.ai_sdk.ui_message import (
    UITextPart,
    UIToolPart,
)
from ai.types import messages as messages_


def test_to_ui_messages_user_and_assistant() -> None:
    msgs = [
        messages_.Message(
            id="u1", role="user", parts=[messages_.TextPart(text="hi")]
        ),
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[messages_.TextPart(text="hello back")],
        ),
    ]
    result = to_ui_messages(msgs)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[1].id == "a1"


def test_to_ui_messages_merges_assistant_tool_internal() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[
                messages_.TextPart(text="calling"),
                messages_.ToolCallPart(
                    tool_call_id="tc1",
                    tool_name="search",
                    tool_args='{"q":"x"}',
                ),
            ],
        ),
        messages_.Message(
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    tool_call_id="tc1",
                    tool_name="search",
                    result={"hits": 2},
                )
            ],
        ),
        messages_.Message(
            role="assistant",
            parts=[messages_.TextPart(text="done")],
        ),
    ]
    result = to_ui_messages(msgs)
    assert len(result) == 1
    ui_msg = result[0]
    assert ui_msg.role == "assistant"
    assert ui_msg.id == "a1"
    assert isinstance(ui_msg.parts[0], UITextPart)
    assert ui_msg.parts[0].text == "calling"
    assert isinstance(ui_msg.parts[1], UIToolPart)
    assert ui_msg.parts[1].state == "output-available"
    assert ui_msg.parts[1].output == {"hits": 2}
    assert isinstance(ui_msg.parts[2], UITextPart)
    assert ui_msg.parts[2].text == "done"


def test_to_ui_messages_internal_role_merges_approval() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[
                messages_.ToolCallPart(
                    tool_call_id="tc1",
                    tool_name="delete",
                    tool_args="{}",
                )
            ],
        ),
        messages_.Message(
            role="internal",
            parts=[
                messages_.HookPart(
                    hook_id="approve_tc1",
                    hook_type="ToolApproval",
                    status="pending",
                )
            ],
        ),
    ]
    result = to_ui_messages(msgs)
    ui_msg = result[0]
    tool_part = ui_msg.parts[0]
    assert isinstance(tool_part, UIToolPart)
    assert tool_part.state == "approval-requested"
    assert tool_part.approval is not None
    assert tool_part.approval.id == "approve_tc1"


def test_to_ui_messages_user_message_uses_own_id() -> None:
    msgs = [
        messages_.Message(
            id="u1", role="user", parts=[messages_.TextPart(text="a")]
        )
    ]
    result = to_ui_messages(msgs)
    assert result[0].id == "u1"


def test_to_ui_messages_uses_first_assistant_id_as_bubble_id() -> None:
    msgs = [
        messages_.Message(
            id="a1",
            role="assistant",
            parts=[messages_.TextPart(text="first")],
        ),
        messages_.Message(
            id="a2",
            role="assistant",
            parts=[messages_.TextPart(text="second")],
        ),
    ]
    result = to_ui_messages(msgs)
    assert len(result) == 1
    assert result[0].id == "a1"
