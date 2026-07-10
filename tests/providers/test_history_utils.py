"""Tests for the provider message-history utilities."""

from __future__ import annotations

import pytest

from ai.providers import history_utils
from ai.types import builders, messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_with_tool_call(
    tool_call_id: str = "tc-1",
    tool_name: str = "calc",
    tool_args: str = '{"x": 1}',
) -> messages.Message:
    return messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_args=tool_args,
            )
        ],
    )


def _tool_result(
    tool_call_id: str = "tc-1",
    tool_name: str = "calc",
    result: str = "42",
) -> messages.Message:
    return builders.tool_message(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=result,
    )


def _internal_message() -> messages.Message:
    return messages.Message(
        role="internal", parts=[messages.TextPart(text="internal")]
    )


def _hook_part() -> messages.HookPart[None]:
    return messages.HookPart(
        hook_id="h1", hook_type="confirm", status="resolved"
    )


def _kinds(issues: list[history_utils.Issue]) -> set[str]:
    return {issue.kind for issue in issues}


def _assert_repair_raises(
    msgs: list[messages.Message], kind: history_utils.IssueKind
) -> None:
    with pytest.raises(history_utils.IntegrityError) as exc_info:
        history_utils.repair(msgs)
    assert kind in _kinds(exc_info.value.issues)


# ---------------------------------------------------------------------------
# repair: clean passthrough
# ---------------------------------------------------------------------------


def test_clean_messages_pass_through() -> None:
    msgs = [
        builders.user_message("hello"),
        builders.assistant_message("world"),
    ]
    result = history_utils.repair(msgs)
    assert len(result) == 2
    assert result[0].text == "hello"
    assert result[1].text == "world"


def test_repair_idempotent() -> None:
    msgs = [
        builders.user_message("hi"),
        _assistant_with_tool_call(),
        _tool_result(),
        builders.assistant_message("done"),
    ]
    once = history_utils.repair(msgs)
    twice = history_utils.repair(once)
    assert len(once) == len(twice)
    for a, b in zip(once, twice, strict=True):
        assert a.role == b.role
        assert len(a.parts) == len(b.parts)


def test_complete_tool_flow_unchanged() -> None:
    """A properly paired tool flow passes through without modification."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        _tool_result(),
        builders.assistant_message("The answer is 4"),
    ]
    result = history_utils.repair(msgs)
    assert len(result) == 4
    assert [m.role for m in result] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_repair_does_not_mutate_input() -> None:
    original = [
        builders.user_message("hi"),
        _assistant_with_tool_call(),
    ]
    original_len = len(original)
    _ = history_utils.repair(original)
    assert len(original) == original_len


# ---------------------------------------------------------------------------
# drop_internal
# ---------------------------------------------------------------------------


def test_drops_internal_messages() -> None:
    internal = _internal_message()
    msgs = [
        builders.user_message("hi"),
        internal,
        builders.assistant_message("hello"),
    ]
    result, issues = history_utils.drop_internal(msgs)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert issues == [
        history_utils.Issue(kind="internal-message", message_id=internal.id)
    ]


def test_strips_internal_parts() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[messages.TextPart(text="hi"), _hook_part()],
    )
    result, issues = history_utils.drop_internal([msg])
    assert len(result) == 1
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], messages.TextPart)
    assert _kinds(issues) == {"internal-part"}


def test_strips_internal_parts_drops_empty_message() -> None:
    """Message with only internal parts becomes empty and is dropped."""
    msg = messages.Message(role="assistant", parts=[_hook_part()])
    result, issues = history_utils.drop_internal([msg])
    assert len(result) == 0
    assert _kinds(issues) == {"internal-part"}


# ---------------------------------------------------------------------------
# fix_tool_args
# ---------------------------------------------------------------------------


def test_fixes_invalid_tool_args() -> None:
    msg = _assistant_with_tool_call(tool_args="not json {{{")
    result, issues = history_utils.fix_tool_args([msg])
    tc = result[0].parts[0]
    assert isinstance(tc, messages.ToolCallPart)
    assert tc.tool_args == "{}"
    assert issues == [
        history_utils.Issue(kind="invalid-tool-args", message_id=msg.id)
    ]


def test_preserves_valid_tool_args() -> None:
    msg = _assistant_with_tool_call(tool_args='{"key": "value"}')
    result, issues = history_utils.fix_tool_args([msg])
    tc = result[0].parts[0]
    assert isinstance(tc, messages.ToolCallPart)
    assert tc.tool_args == '{"key": "value"}'
    assert issues == []


# ---------------------------------------------------------------------------
# close_orphaned_tool_calls
# ---------------------------------------------------------------------------


def test_inserts_synthetic_result_for_orphaned_call_at_end() -> None:
    """Tool call at end of history with no result gets a synthetic one."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
    ]
    result, issues = history_utils.close_orphaned_tool_calls(msgs)
    assert len(result) == 3
    assert result[2].role == "tool"
    tr = result[2].tool_results[0]
    assert tr.tool_call_id == "tc-1"
    assert tr.is_error is True
    assert _kinds(issues) == {"orphaned-tool-call"}


def test_inserts_synthetic_result_before_user_interruption() -> None:
    """User message interrupting tool flow triggers synthetic results."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        builders.user_message("never mind"),
    ]
    result, _ = history_utils.close_orphaned_tool_calls(msgs)
    assert len(result) == 4
    # Synthetic result inserted before the user message.
    assert result[2].role == "tool"
    assert result[2].tool_results[0].is_error is True
    assert result[3].role == "user"
    assert result[3].text == "never mind"


def test_inserts_synthetic_result_before_next_assistant() -> None:
    """New assistant message with pending tool calls adds synthetic results."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        builders.assistant_message("actually, the answer is 4"),
    ]
    result, _ = history_utils.close_orphaned_tool_calls(msgs)
    assert len(result) == 4
    assert result[2].role == "tool"
    assert result[2].tool_results[0].is_error is True
    assert result[3].role == "assistant"


def test_multiple_orphaned_calls_get_individual_results() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id="tc-1", tool_name="a", tool_args="{}"
            ),
            messages.ToolCallPart(
                tool_call_id="tc-2", tool_name="b", tool_args="{}"
            ),
        ],
    )
    result, _ = history_utils.close_orphaned_tool_calls(
        [builders.user_message("go"), msg]
    )
    # Synthetic tool message should have results for both calls.
    synthetic = result[2]
    assert synthetic.role == "tool"
    ids = {tr.tool_call_id for tr in synthetic.tool_results}
    assert ids == {"tc-1", "tc-2"}


def test_partial_results_only_fills_missing() -> None:
    """If some results exist, only the missing ones get synthetic fills."""
    msgs = [
        builders.user_message("go"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="a", tool_args="{}"
                ),
                messages.ToolCallPart(
                    tool_call_id="tc-2", tool_name="b", tool_args="{}"
                ),
            ],
        ),
        _tool_result(tool_call_id="tc-1"),
        # tc-2 is missing, then user interrupts
        builders.user_message("stop"),
    ]
    result, _ = history_utils.close_orphaned_tool_calls(msgs)
    assert len(result) == 5
    synthetic = result[3]
    assert synthetic.role == "tool"
    assert len(synthetic.tool_results) == 1
    assert synthetic.tool_results[0].tool_call_id == "tc-2"
    assert synthetic.tool_results[0].is_error is True


def test_repair_closes_orphaned_tool_calls() -> None:
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
    ]
    result = history_utils.repair(msgs)
    assert len(result) == 3
    assert result[2].role == "tool"
    assert result[2].tool_results[0].is_error is True


# ---------------------------------------------------------------------------
# repair: unfixable issues raise
# ---------------------------------------------------------------------------


def test_orphaned_tool_result_raises() -> None:
    """Tool result referencing a nonexistent call raises."""
    msgs = [
        builders.user_message("hi"),
        _tool_result(tool_call_id="nonexistent"),
    ]
    _assert_repair_raises(msgs, "orphaned-tool-result")


def test_out_of_sequence_tool_result_raises() -> None:
    """A late tool result cannot arrive after another conversation turn."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
        builders.user_message("never mind"),
        _tool_result(),
    ]
    _assert_repair_raises(msgs, "orphaned-tool-result")


def test_duplicate_tool_calls_raise() -> None:
    """Two assistant messages using the same tool_call_id raise."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1", tool_args='{"v": 1}'),
        _tool_result(tool_call_id="tc-1", result="old"),
        _assistant_with_tool_call(tool_call_id="tc-1", tool_args='{"v": 2}'),
        _tool_result(tool_call_id="tc-1", result="new"),
    ]
    with pytest.raises(history_utils.IntegrityError) as exc_info:
        history_utils.repair(msgs)
    kinds = _kinds(exc_info.value.issues)
    assert "duplicate-tool-call" in kinds
    assert "duplicate-tool-result" in kinds


def test_duplicate_tool_calls_within_same_message_raise() -> None:
    """Two tool calls with the same ID in one assistant message raise."""
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id="tc-1", tool_name="a", tool_args='{"v": 1}'
            ),
            messages.ToolCallPart(
                tool_call_id="tc-1", tool_name="a", tool_args='{"v": 2}'
            ),
        ],
    )
    _assert_repair_raises(
        [builders.user_message("go"), msg], "duplicate-tool-call"
    )


def test_duplicate_tool_results_raise() -> None:
    """Two tool messages with results for the same call raise."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1"),
        _tool_result(tool_call_id="tc-1", result="first"),
        _tool_result(tool_call_id="tc-1", result="second"),
    ]
    _assert_repair_raises(msgs, "duplicate-tool-result")


def test_duplicate_tool_results_within_same_message_raise() -> None:
    """Two results for the same call ID in one tool message raise."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1"),
        messages.Message(
            role="tool",
            parts=[
                builders.tool_result_part("tc-1", result="first"),
                builders.tool_result_part("tc-1", result="second"),
            ],
        ),
    ]
    _assert_repair_raises(msgs, "duplicate-tool-result")


# ---------------------------------------------------------------------------
# check_tool_ids
# ---------------------------------------------------------------------------


def test_check_tool_ids_clean_history() -> None:
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
        _tool_result(),
    ]
    assert history_utils.check_tool_ids(msgs) == []


def test_check_tool_ids_reports_without_raising() -> None:
    msgs = [
        builders.user_message("hi"),
        _tool_result(tool_call_id="nonexistent"),
    ]
    assert _kinds(history_utils.check_tool_ids(msgs)) == {
        "orphaned-tool-result"
    }


# ---------------------------------------------------------------------------
# inspect / validate
# ---------------------------------------------------------------------------


def test_inspect_clean_history_returns_nothing() -> None:
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        _tool_result(),
        builders.assistant_message("The answer is 4"),
    ]
    assert history_utils.inspect(msgs) == []


def test_inspect_collects_all_issues() -> None:
    msgs = [
        _internal_message(),
        builders.user_message("go"),
        messages.Message(
            role="assistant",
            parts=[
                _hook_part(),
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="a", tool_args="broken"
                ),
            ],
        ),
    ]
    kinds = _kinds(history_utils.inspect(msgs))
    assert kinds == {
        "internal-message",
        "internal-part",
        "invalid-tool-args",
        "orphaned-tool-call",
    }


def test_inspect_reports_issue_message_ids() -> None:
    internal = _internal_message()
    issues = history_utils.inspect([internal, builders.user_message("hi")])
    assert issues == [
        history_utils.Issue(kind="internal-message", message_id=internal.id)
    ]


def test_inspect_keeps_recoverable_issues_when_history_is_corrupt() -> None:
    msgs = [
        _internal_message(),
        builders.user_message("go"),
        messages.Message(
            role="assistant",
            parts=[
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="a", tool_args="{}"
                ),
                messages.ToolCallPart(
                    tool_call_id="tc-1", tool_name="b", tool_args="{}"
                ),
            ],
        ),
    ]
    kinds = _kinds(history_utils.inspect(msgs))
    assert "internal-message" in kinds
    assert "duplicate-tool-call" in kinds


def test_inspect_does_not_change_history() -> None:
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
    ]
    _ = history_utils.inspect(msgs)
    assert len(msgs) == 2


def test_validate_raises_on_any_issue() -> None:
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
    ]
    with pytest.raises(history_utils.IntegrityError) as exc_info:
        history_utils.validate(msgs)
    assert "orphaned-tool-call" in _kinds(exc_info.value.issues)


def test_validate_passes_clean_history() -> None:
    history_utils.validate(
        [
            builders.user_message("hi"),
            builders.assistant_message("hello"),
        ]
    )
