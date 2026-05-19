from __future__ import annotations

from typing import Any

import pytest

from ai.agents.agent import MessageBundle
from ai.agents.ui.ai_sdk import to_messages
from ai.agents.ui.ai_sdk.inbound import (
    _normalize_ui_messages,
    extract_approvals,
)
from ai.agents.ui.ai_sdk.ui_message import UIMessage, UIToolPart
from ai.types import messages as messages_


def _ui(role: str, *parts: dict[str, Any], id: str = "m1") -> UIMessage:
    return UIMessage.model_validate(
        {"id": id, "role": role, "parts": list(parts)}
    )


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
    messages, approvals = to_messages([_ui("user", _text("hello"))])
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].text == "hello"
    assert approvals == []


def test_to_messages_splits_at_tool_boundary() -> None:
    messages, _ = to_messages(
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
    assert [m.role for m in messages] == ["assistant", "tool", "assistant"]
    assert messages[1].tool_results[0].tool_call_id == "tc1"


def test_to_messages_keeps_pending_approval_tombstone() -> None:
    """Pending approvals carry no response — leave the tombstone in history."""
    messages, _ = to_messages(
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
    )
    assert [m.role for m in messages] == ["assistant", "internal"]
    hook_part = messages[1].parts[0]
    assert isinstance(hook_part, messages_.HookPart)
    assert hook_part.hook_type == "ToolApproval"
    assert hook_part.status == "pending"


def test_to_messages_drops_resolved_approval_tombstone() -> None:
    """Resolved approvals come back via the side-channel."""
    messages, approvals = to_messages(
        [
            _ui(
                "assistant",
                _tool(
                    "delete",
                    "tc1",
                    "approval-responded",
                    approval={
                        "id": "approve_tc1",
                        "approved": True,
                        "reason": None,
                    },
                ),
            )
        ],
    )
    assert [m.role for m in messages] == ["assistant"]
    assert [a.hook_id for a in approvals] == ["approve_tc1"]


def test_to_messages_keeps_trailing_assistant_when_approved() -> None:
    messages, approvals = to_messages(
        [
            _ui("user", _text("delete it"), id="u1"),
            _ui(
                "assistant",
                _tool(
                    "delete",
                    "tc1",
                    "approval-responded",
                    approval={
                        "id": "approve_tc1",
                        "approved": True,
                        "reason": None,
                    },
                ),
                id="a1",
            ),
        ],
    )
    assert [m.role for m in messages] == ["user", "assistant"]
    assert [a.hook_id for a in approvals] == ["approve_tc1"]


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


def test_to_messages_decodes_subagent_tool_output() -> None:
    """A sub-agent tool's wire UIMessage decodes back to MessageBundle.

    ``result`` carries the rich MessageBundle so a subsequent UI render
    gets the same shape we sent.  ``model_input`` is left unset here —
    populating it requires the tool registry, which lives in
    :meth:`Agent.run`.
    """
    # Wire shape: tool-_research_tool with output = UIMessage{parts=[text]}.
    ui = [
        _ui("user", _text("research mars"), id="u1"),
        _ui(
            "assistant",
            _tool(
                "_research_tool",
                "tc1",
                "output-available",
                input={"topic": "mars"},
                output={
                    "id": "sub-1",
                    "role": "assistant",
                    "parts": [{"type": "text", "text": "Mars has two moons."}],
                },
            ),
            id="a1",
        ),
    ]
    messages, _ = to_messages(ui)

    # Find the tool message with the decoded result.
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    result_part = tool_msgs[0].tool_results[0]
    assert isinstance(result_part.result, MessageBundle)
    assert not result_part.has_model_input


def test_to_messages_passthrough_keeps_wire_shape() -> None:
    """Non-UIMessage tool outputs stay in their wire form."""
    ui = [
        _ui("user", _text("hi"), id="u1"),
        _ui(
            "assistant",
            _tool(
                "ping",
                "tc1",
                "output-available",
                input={},
                output={"pong": True},
            ),
            id="a1",
        ),
    ]
    messages, _ = to_messages(ui)
    tool_msgs = [m for m in messages if m.role == "tool"]
    part = tool_msgs[0].tool_results[0]
    assert part.result == {"pong": True}
    assert part.get_model_input() == {"pong": True}
