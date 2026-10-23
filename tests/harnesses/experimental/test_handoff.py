"""The rendering rule, pinned without a network.

Text verbatim; tool calls as native items with the ORIGINAL name and
arguments; tool results as strings bound to their call; reasoning as a
user-role note after the turn it belonged to. Both renderers, same rule.
"""

from __future__ import annotations

import json

from ai.harnesses.experimental._handoff import (
    NOTE_PREFIX,
    reasoning_count,
    to_claude_records,
    to_codex_items,
)
from ai.types.messages import (
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

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


def test_codex_items_follow_the_rule() -> None:
    items = to_codex_items(HISTORY)
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
    assert (
        call["name"] == "Write"
        and json.loads(call["arguments"])["content"] == "two eggs"
    )
    assert call["call_id"] == "call_1"
    out = items[4]
    assert out["call_id"] == "call_1" and out["output"] == json.dumps(
        {"ok": True}
    )
    note = items[3]["content"][0]["text"]
    assert note.startswith(NOTE_PREFIX) and "Write is the tool" in note


def test_claude_records_follow_the_rule() -> None:
    recs = to_claude_records(HISTORY, "sid-1", "/w")
    roles = [r["type"] for r in recs]
    # user prompt, assistant (text + tool_use), tool result, assistant — NO
    # reasoning note: Claude's safeguards refuse replayed reasoning.
    assert roles == ["user", "assistant", "user", "assistant"]
    assistant = recs[1]["message"]["content"]
    assert [b["type"] for b in assistant] == ["text", "tool_use"]
    assert (
        assistant[1]["name"] == "Write"
        and assistant[1]["input"]["content"] == "two eggs"
    )
    assert assistant[1]["id"] == "call_1"
    result = recs[2]["message"]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "call_1"
    assert not any(NOTE_PREFIX in str(r["message"]["content"]) for r in recs)
    assert reasoning_count(HISTORY) == 1  # what the adapter warns about
    # a real chain, in the CLI's own store shape
    assert recs[0]["parentUuid"] is None
    assert all(
        recs[i]["parentUuid"] == recs[i - 1]["uuid"]
        for i in range(1, len(recs))
    )
    assert all(r["sessionId"] == "sid-1" and r["cwd"] == "/w" for r in recs)


def test_reasoning_is_never_the_agents_own_thought() -> None:
    """No thinking block, no reasoning item: both were measured to fail —
    one rejected, one silently ignored. It travels as a note, or not at all."""
    assert not any(i["type"] == "reasoning" for i in to_codex_items(HISTORY))
    blocks = [
        b
        for r in to_claude_records(HISTORY, "s", "/w")
        if isinstance(r["message"]["content"], list)
        for b in r["message"]["content"]
    ]
    assert not any(b["type"] == "thinking" for b in blocks)


def test_empty_parts_produce_nothing() -> None:
    assert (
        to_codex_items(
            [Message(role="assistant", parts=[TextPart(text="   ")])]
        )
        == []
    )
    assert (
        to_claude_records(
            [Message(role="assistant", parts=[ReasoningPart(text="")])],
            "s",
            "/w",
        )
        == []
    )


def test_tool_results_are_strings_for_codex() -> None:
    items = to_codex_items(
        [
            Message(
                role="tool",
                parts=[
                    ToolResultPart(
                        tool_call_id="c", tool_name="x", result=[1, 2]
                    )
                ],
            )
        ]
    )
    assert items[0]["output"] == "[1, 2]"
