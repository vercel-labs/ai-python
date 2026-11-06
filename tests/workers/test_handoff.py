"""The seeding rule, pinned without a CLI.

Text verbatim; tool calls as native items with the ORIGINAL name and
arguments; tool results as strings bound to their call; reasoning as a
user-role note after the turn it belonged to (codex), or not at all
(claude, pi). A worker holds a tool an `ai.Agent` ran as a call and a
`tool` message, and a tool the CLI ran as built-in parts in one message:
each renderer takes both, and both seed the same.
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
from ai.workers import _handoff
from ai.workers.claude_code import handoff as claude_handoff
from ai.workers.codex import handoff as codex_handoff
from ai.workers.pi import handoff as pi_handoff

# Write as a tool an `ai.Agent` ran: its result in a `tool` message.
AGENT_TOOL = [
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

# The same conversation, Write as a tool the CLI ran: built-in parts.
CLI_TOOL = [
    AGENT_TOOL[0],
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
    AGENT_TOOL[3],
]


def test_codex_items_follow_the_rule() -> None:
    items = codex_handoff.to_codex_items(AGENT_TOOL)
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
    assert note.startswith(codex_handoff.NOTE_PREFIX)
    assert "Write is the tool" in note


def test_both_shapes_seed_the_same() -> None:
    assert codex_handoff.to_codex_items(CLI_TOOL) == (
        codex_handoff.to_codex_items(AGENT_TOOL)
    )
    assert pi_handoff.to_pi_messages(CLI_TOOL, timestamp=1) == (
        pi_handoff.to_pi_messages(AGENT_TOOL, timestamp=1)
    )
    assert [
        (r["type"], r["message"])
        for r in claude_handoff.to_claude_records(CLI_TOOL, "s", "/w")
    ] == [
        (r["type"], r["message"])
        for r in claude_handoff.to_claude_records(AGENT_TOOL, "s", "/w")
    ]


def test_claude_records_follow_the_rule() -> None:
    recs = claude_handoff.to_claude_records(CLI_TOOL, "sid-1", "/w")
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
    assert _handoff.reasoning_count(CLI_TOOL) == 1  # what is warned about
    # a real chain, in the CLI's own store shape
    assert recs[0]["parentUuid"] is None
    assert all(
        recs[i]["parentUuid"] == recs[i - 1]["uuid"]
        for i in range(1, len(recs))
    )
    assert all(r["sessionId"] == "sid-1" and r["cwd"] == "/w" for r in recs)


def test_pi_merges_assistant_messages_and_keeps_tool_results() -> None:
    rendered = pi_handoff.to_pi_messages(
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


def test_text_after_a_cli_tool_stays_after_its_result() -> None:
    """Claude and pi keep a result in a message of its own: the message
    with the call ends there, and what follows comes after the result."""
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
    recs = claude_handoff.to_claude_records([one], "s", "/w")
    assert [r["type"] for r in recs] == ["assistant", "user", "assistant"]
    assert recs[1]["message"]["content"][0]["type"] == "tool_result"
    assert recs[2]["message"]["content"] == [
        {"type": "text", "text": "It says x."}
    ]
    rendered = pi_handoff.to_pi_messages([one], timestamp=1)
    assert [r["role"] for r in rendered] == [
        "assistant",
        "toolResult",
        "assistant",
    ]
    assert rendered[2]["content"] == [{"type": "text", "text": "It says x."}]
    items = codex_handoff.to_codex_items([one])
    assert [i["type"] for i in items] == [
        "function_call",
        "function_call_output",
        "message",
    ]


def test_reasoning_is_never_the_agents_own_thought() -> None:
    """No thinking block, no reasoning item: both were measured to fail,
    one rejected, one silently ignored. It travels as a note, or not at all."""
    items = codex_handoff.to_codex_items(AGENT_TOOL)
    assert not any(i["type"] == "reasoning" for i in items)
    blocks = [
        b
        for r in claude_handoff.to_claude_records(AGENT_TOOL, "s", "/w")
        if isinstance(r["message"]["content"], list)
        for b in r["message"]["content"]
    ]
    assert not any(b["type"] == "thinking" for b in blocks)


def test_empty_parts_produce_nothing() -> None:
    blank = [Message(role="assistant", parts=[TextPart(text="   ")])]
    assert codex_handoff.to_codex_items(blank) == []
    empty = [Message(role="assistant", parts=[ReasoningPart(text="")])]
    assert claude_handoff.to_claude_records(empty, "s", "/w") == []
