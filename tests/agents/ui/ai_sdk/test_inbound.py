from __future__ import annotations

from typing import Any

import pytest

from ai.agents.ui.ai_sdk import to_messages
from ai.agents.ui.ai_sdk.inbound import (
    _normalize_ui_messages,
    extract_approvals,
)
from ai.agents.ui.ai_sdk.ui_message import UIMessage, UIToolPart


def _ui(role: str, *parts: dict[str, Any], id: str = "m1") -> UIMessage:
    return UIMessage.model_validate({"id": id, "role": role, "parts": list(parts)})


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


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


def test_to_messages_user_text() -> None:
    result = to_messages([_ui("user", _text("hello"))])
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].text == "hello"


def test_to_messages_splits_at_tool_boundary() -> None:
    result = to_messages(
        [
            _ui(
                "assistant",
                _text("before"),
                _tool(
                    "search",
                    "tc1",
                    "output-available",
                    input={"q": "x"},
                    output={"hits": 3},
                ),
                _text("after"),
            )
        ]
    )
    assert [m.role for m in result] == ["assistant", "tool", "assistant"]
    assert result[1].tool_results[0].tool_call_id == "tc1"


def test_to_messages_approval_hook_emitted_as_internal() -> None:
    result = to_messages(
        [
            _ui(
                "assistant",
                _tool(
                    "delete",
                    "tc1",
                    "approval-requested",
                    approval={"id": "approve_tc1"},
                ),
            )
        ],
        apply_approvals_=False,
    )
    assert [m.role for m in result] == ["assistant", "internal"]
    hook = result[1].parts[0]
    assert hook.type == "hook"
    assert hook.hook_id == "approve_tc1"


def test_to_messages_strips_trailing_assistant_when_approved() -> None:
    result = to_messages(
        [
            _ui("user", _text("delete it"), id="u1"),
            _ui(
                "assistant",
                _tool(
                    "delete",
                    "tc1",
                    "approval-responded",
                    approval={"id": "approve_tc1", "approved": True, "reason": None},
                ),
                id="a1",
            ),
        ],
        apply_approvals_=False,
    )
    assert [m.role for m in result] == ["user", "internal"]


def test_extract_approvals_returns_approved_responses() -> None:
    approvals = extract_approvals(
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
    assert len(approvals) == 1
    assert approvals[0].hook_id == "approve_tc1"
    assert approvals[0].granted is False
    assert approvals[0].reason == "nope"


def test_normalize_ui_messages_heals_stale_tool_state() -> None:
    ui = [
        _ui(
            "assistant",
            _tool("x", "tc1", "input-available", output={"ok": True}),
        )
    ]
    normalized = _normalize_ui_messages(ui)
    tool_part = normalized[0].parts[0]
    assert isinstance(tool_part, UIToolPart)
    assert tool_part.state == "output-available"


def test_to_messages_rejects_empty_user() -> None:
    ui = [UIMessage.model_validate({"id": "u1", "role": "user", "parts": []})]
    with pytest.raises(ValueError):
        to_messages(ui)
