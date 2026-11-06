"""The seeding rule, pinned without a CLI.

Text verbatim; tool calls as native items with the ORIGINAL name and
arguments; tool results as strings bound to their call; reasoning as a
user-role note after the turn it belonged to (codex), or not at all
(claude, pi). The CLI's own tools arrive as built-in parts.
"""

from __future__ import annotations

import json

from ai.types.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from ai.workers._adapters import _handoff

HISTORY = [
    Message(role="user", parts=[TextPart(text="Create recipe.txt")]),
    Message(
        role="assistant",
        parts=[
            ReasoningPart(text="The user wants a file; Write is the tool."),
            TextPart(text="Creating it."),
            ToolCallPart(
                tool_call_id="call_1",
                tool_name="Write",
                tool_args=json.dumps(
                    {"file_path": "/w/recipe.txt", "content": "two eggs"}
                ),
            ),
        ],
    ),
    Message(
        role="tool",
        parts=[
            ToolResultPart(
                tool_call_id="call_1",
                tool_name="Write",
                result={"ok": True},
                result_kind="json",
            )
        ],
    ),
    Message(role="assistant", parts=[TextPart(text="Done.")]),
]

# The same conversation, the way a worker holds it.
MODEL_VIEW = [
    HISTORY[0],
    Message(
        role="assistant",
        parts=[
            ReasoningPart(text="The user wants a file; Write is the tool."),
            TextPart(text="Creating it."),
            BuiltinToolCallPart(
                tool_call_id="call_1",
                tool_name="Write",
                tool_args=json.dumps(
                    {"file_path": "/w/recipe.txt", "content": "two eggs"}
                ),
            ),
            BuiltinToolReturnPart(
                tool_call_id="call_1", tool_name="Write", result={"ok": True}
            ),
        ],
    ),
    HISTORY[3],
]


def test_codex_items_follow_the_rule() -> None:
    items = _handoff.to_codex_items(HISTORY)
    kinds = [(i["type"], i.get("role")) for i in items]
    assert kinds == [
        ("message", "user"),
        ("message", "assistant"),  # "Creating it."
        ("function_call", None),  # Write, name untouched
        ("message", "user"),  # the reasoning note, after its turn
        ("function_call_output", None),
        ("message", "assistant"),  # "Done."
    ]
    call = items[2]
    assert call["name"] == "Write"
    assert json.loads(call["arguments"])["content"] == "two eggs"
    assert call["call_id"] == "call_1"
    out = items[4]
    assert out["call_id"] == "call_1"
    assert out["output"] == json.dumps({"ok": True})
    note = items[3]["content"][0]["text"]
    assert note.startswith(_handoff.NOTE_PREFIX)
    assert "Write is the tool" in note


def test_the_model_view_seeds_the_same_items() -> None:
    assert _handoff.to_codex_items(MODEL_VIEW) == _handoff.to_codex_items(
        HISTORY
    )


def test_claude_records_follow_the_rule() -> None:
    recs = _handoff.to_claude_records(MODEL_VIEW, "sid-1", "/w")
    # user prompt, assistant (text + tool_use), tool result, assistant; NO
    # reasoning note: Claude's safeguards refuse replayed reasoning.
    assert [r["type"] for r in recs] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assistant = recs[1]["message"]["content"]
    assert [b["type"] for b in assistant] == ["text", "tool_use"]
    assert assistant[1]["name"] == "Write"
    assert assistant[1]["input"]["content"] == "two eggs"
    assert assistant[1]["id"] == "call_1"
    result = recs[2]["message"]["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "call_1"
    assert _handoff.reasoning_count(MODEL_VIEW) == 1  # what is warned about
    # a real chain, in the CLI's own store shape
    assert recs[0]["parentUuid"] is None
    assert all(
        recs[i]["parentUuid"] == recs[i - 1]["uuid"]
        for i in range(1, len(recs))
    )
    assert all(r["sessionId"] == "sid-1" and r["cwd"] == "/w" for r in recs)


def test_pi_merges_assistant_messages_and_keeps_tool_results() -> None:
    rendered = _handoff.to_pi_messages(
        [
            Message(role="user", parts=[TextPart(text="hi")]),
            Message(role="assistant", parts=[TextPart(text="one")]),
            Message(
                role="assistant",
                parts=[
                    BuiltinToolCallPart(
                        tool_call_id="c1", tool_name="Bash", tool_args="{}"
                    ),
                    BuiltinToolReturnPart(
                        tool_call_id="c1",
                        tool_name="Bash",
                        result={"ok": True},
                        is_error=True,
                    ),
                ],
            ),
        ],
        timestamp=1,
    )
    assert [r["role"] for r in rendered] == ["user", "assistant", "toolResult"]
    assert [b["type"] for b in rendered[1]["content"]] == ["text", "toolCall"]
    assert rendered[1]["stopReason"] == "toolUse"
    assert rendered[2]["content"] == [{"type": "text", "text": '{"ok": true}'}]
    assert rendered[2]["isError"] is True


def test_model_view_and_harness_view_are_inverse() -> None:
    model = _handoff.model_view(HISTORY)
    assert [m.role for m in model] == ["user", "assistant", "assistant"]
    parts = model[1].parts
    assert isinstance(parts[2], BuiltinToolCallPart)
    assert isinstance(parts[3], BuiltinToolReturnPart)
    assert parts[3].result == {"ok": True}
    back = _handoff.harness_view(model)
    assert [m.role for m in back] == ["user", "assistant", "tool", "assistant"]
    assert isinstance(back[2].parts[0], ToolResultPart)


def test_text_after_a_cli_tool_stays_after_its_result() -> None:
    one = Message(
        role="assistant",
        parts=[
            BuiltinToolCallPart(tool_call_id="c", tool_name="Read"),
            BuiltinToolReturnPart(
                tool_call_id="c", tool_name="Read", result="x"
            ),
            TextPart(text="It says x."),
        ],
    )
    first, result, after = _handoff.harness_view([one])
    assert (first.role, result.role, after.role) == (
        "assistant",
        "tool",
        "assistant",
    )
    assert first.id == one.id and after.id != one.id
    assert after.text == "It says x."


def test_a_stored_result_without_a_name_takes_its_calls() -> None:
    stored = [
        Message(
            role="assistant",
            parts=[
                ToolCallPart(tool_call_id="c", tool_name="Read", tool_args="")
            ],
        ),
        Message(
            role="tool",
            parts=[ToolResultPart(tool_call_id="c", tool_name="", result="x")],
        ),
    ]
    (message,) = _handoff.model_view(stored)
    result = message.parts[1]
    assert isinstance(result, BuiltinToolReturnPart)
    assert result.tool_name == "Read"


def test_reasoning_is_never_the_agents_own_thought() -> None:
    """No thinking block, no reasoning item: both were measured to fail,
    one rejected, one silently ignored. It travels as a note, or not at all."""
    items = _handoff.to_codex_items(HISTORY)
    assert not any(i["type"] == "reasoning" for i in items)
    blocks = [
        b
        for r in _handoff.to_claude_records(HISTORY, "s", "/w")
        if isinstance(r["message"]["content"], list)
        for b in r["message"]["content"]
    ]
    assert not any(b["type"] == "thinking" for b in blocks)


def test_empty_parts_produce_nothing() -> None:
    blank = [Message(role="assistant", parts=[TextPart(text="   ")])]
    assert _handoff.to_codex_items(blank) == []
    empty = [Message(role="assistant", parts=[ReasoningPart(text="")])]
    assert _handoff.to_claude_records(empty, "s", "/w") == []
