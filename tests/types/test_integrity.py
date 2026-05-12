"""Tests for message integrity checker."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal
from unittest.mock import patch

import pydantic
import pytest

import ai
from ai import models
from ai.types import builders, messages
from ai.types import events as events_
from ai.types.integrity import IntegrityError, prepare_messages

from ..conftest import MOCK_MODEL, mock_generate, mock_llm, text_msg

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


def _assert_raises_issue(
    msgs: list[messages.Message],
    issue: str,
    *,
    mode: Literal["auto", "strict"] = "auto",
) -> None:
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode=mode)
    assert issue in exc_info.value.issues


# ---------------------------------------------------------------------------
# Clean passthrough
# ---------------------------------------------------------------------------


def test_clean_messages_pass_through() -> None:
    msgs = [
        builders.user_message("hello"),
        builders.assistant_message("world"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 2
    assert result[0].text == "hello"
    assert result[1].text == "world"


def test_idempotent() -> None:
    msgs = [
        builders.user_message("hi"),
        _assistant_with_tool_call(),
        _tool_result(),
        builders.assistant_message("done"),
    ]
    once = prepare_messages(msgs)
    twice = prepare_messages(once)
    assert len(once) == len(twice)
    for a, b in zip(once, twice, strict=True):
        assert a.role == b.role
        assert len(a.parts) == len(b.parts)


# ---------------------------------------------------------------------------
# Internal messages
# ---------------------------------------------------------------------------


def test_drops_internal_messages() -> None:
    msgs = [
        builders.user_message("hi"),
        messages.Message(role="internal", parts=[messages.TextPart(text="internal")]),
        builders.assistant_message("hello"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"


def test_internal_strict_raises() -> None:
    msgs = [
        messages.Message(role="internal", parts=[messages.TextPart(text="x")]),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    assert "internal-message" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Internal parts (HookPart)
# ---------------------------------------------------------------------------


def test_strips_internal_parts() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.TextPart(text="hi"),
            messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
        ],
    )
    result = prepare_messages([msg])
    assert len(result) == 1
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], messages.TextPart)


def test_strips_internal_parts_drops_empty_message() -> None:
    """Message with only internal parts becomes empty and is dropped."""
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
        ],
    )
    result = prepare_messages([msg])
    assert len(result) == 0


def test_internal_parts_strict_raises() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
        ],
    )
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages([msg], mode="strict")
    assert "internal-part" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Invalid tool args
# ---------------------------------------------------------------------------


def test_fixes_invalid_tool_args() -> None:
    msg = _assistant_with_tool_call(tool_args="not json {{{")
    result = prepare_messages([msg])
    tc = result[0].parts[0]
    assert isinstance(tc, messages.ToolCallPart)
    assert tc.tool_args == "{}"


def test_preserves_valid_tool_args() -> None:
    msg = _assistant_with_tool_call(tool_args='{"key": "value"}')
    result = prepare_messages([msg])
    tc = result[0].parts[0]
    assert isinstance(tc, messages.ToolCallPart)
    assert tc.tool_args == '{"key": "value"}'


def test_invalid_tool_args_strict_raises() -> None:
    msg = _assistant_with_tool_call(tool_args="broken")
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages([msg], mode="strict")
    assert "invalid-tool-args" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Orphaned tool calls (no matching result) — auto-fixable
# ---------------------------------------------------------------------------


def test_inserts_synthetic_result_for_orphaned_call_at_end() -> None:
    """Tool call at end of history with no result gets a synthetic one."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 3
    assert result[2].role == "tool"
    tr = result[2].tool_results[0]
    assert tr.tool_call_id == "tc-1"
    assert tr.is_error is True


def test_inserts_synthetic_result_before_user_interruption() -> None:
    """User message interrupting tool flow triggers synthetic results."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        builders.user_message("never mind"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 4
    # Synthetic result inserted before the user message.
    assert result[2].role == "tool"
    assert result[2].tool_results[0].is_error is True
    assert result[3].role == "user"
    assert result[3].text == "never mind"


def test_inserts_synthetic_result_before_next_assistant() -> None:
    """New assistant message while tool calls pending triggers synthetic results."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        builders.assistant_message("actually, the answer is 4"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 4
    assert result[2].role == "tool"
    assert result[2].tool_results[0].is_error is True
    assert result[3].role == "assistant"


def test_multiple_orphaned_calls_get_individual_results() -> None:
    msg = messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(tool_call_id="tc-1", tool_name="a", tool_args="{}"),
            messages.ToolCallPart(tool_call_id="tc-2", tool_name="b", tool_args="{}"),
        ],
    )
    result = prepare_messages([builders.user_message("go"), msg])
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
    result = prepare_messages(msgs)
    # user, assistant, tool(tc-1), synthetic-tool(tc-2), user
    assert len(result) == 5
    synthetic = result[3]
    assert synthetic.role == "tool"
    assert len(synthetic.tool_results) == 1
    assert synthetic.tool_results[0].tool_call_id == "tc-2"
    assert synthetic.tool_results[0].is_error is True


def test_orphaned_tool_call_strict_raises() -> None:
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    assert "orphaned-tool-call" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Orphaned tool results (no matching call) — always raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "strict"])
def test_orphaned_tool_result_always_raises(
    mode: Literal["auto", "strict"],
) -> None:
    """Tool result referencing a nonexistent call always raises."""
    msgs = [
        builders.user_message("hi"),
        _tool_result(tool_call_id="nonexistent"),
    ]
    _assert_raises_issue(msgs, "orphaned-tool-result", mode=mode)


def test_out_of_sequence_tool_result_raises() -> None:
    """A late tool result cannot arrive after another conversation turn."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(),
        builders.user_message("never mind"),
        _tool_result(),
    ]
    _assert_raises_issue(msgs, "orphaned-tool-result")


# ---------------------------------------------------------------------------
# Complete tool flow (no issues)
# ---------------------------------------------------------------------------


def test_complete_tool_flow_unchanged() -> None:
    """A properly paired tool flow passes through without modification."""
    msgs = [
        builders.user_message("calc 2+2"),
        _assistant_with_tool_call(),
        _tool_result(),
        builders.assistant_message("The answer is 4"),
    ]
    result = prepare_messages(msgs)
    assert len(result) == 4
    assert [m.role for m in result] == ["user", "assistant", "tool", "assistant"]


# ---------------------------------------------------------------------------
# Strict mode collects multiple issues
# ---------------------------------------------------------------------------


def test_strict_collects_all_issues() -> None:
    msgs = [
        messages.Message(role="internal", parts=[messages.TextPart(text="x")]),
        messages.Message(
            role="assistant",
            parts=[
                messages.TextPart(text="hi"),
                messages.HookPart(hook_id="h1", hook_type="confirm", status="resolved"),
            ],
        ),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    issues = exc_info.value.issues
    assert "internal-message" in issues
    assert "internal-part" in issues


def test_strict_keeps_recoverable_issues_when_history_is_corrupt() -> None:
    msgs = [
        messages.Message(role="internal", parts=[messages.TextPart(text="x")]),
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
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs, mode="strict")
    assert "internal-message" in exc_info.value.issues
    assert "duplicate-tool-call" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Duplicate tool call IDs — always raises
# ---------------------------------------------------------------------------


def test_duplicate_tool_calls_raises_in_auto() -> None:
    """Two assistant messages using the same tool_call_id always raises."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1", tool_args='{"v": 1}'),
        _tool_result(tool_call_id="tc-1", result="old"),
        _assistant_with_tool_call(tool_call_id="tc-1", tool_args='{"v": 2}'),
        _tool_result(tool_call_id="tc-1", result="new"),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs)
    assert "duplicate-tool-call" in exc_info.value.issues
    assert "duplicate-tool-result" in exc_info.value.issues


def test_duplicate_tool_calls_within_same_message_raises() -> None:
    """Two tool calls with the same ID in one assistant message raises."""
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
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages([builders.user_message("go"), msg])
    assert "duplicate-tool-call" in exc_info.value.issues


def test_duplicate_tool_results_raises_in_auto() -> None:
    """Two tool messages with results for the same call always raises."""
    msgs = [
        builders.user_message("go"),
        _assistant_with_tool_call(tool_call_id="tc-1"),
        _tool_result(tool_call_id="tc-1", result="first"),
        _tool_result(tool_call_id="tc-1", result="second"),
    ]
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs)
    assert "duplicate-tool-result" in exc_info.value.issues


def test_duplicate_tool_results_within_same_message_raises() -> None:
    """Two results for the same call ID in one tool message raises."""
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
    with pytest.raises(IntegrityError) as exc_info:
        prepare_messages(msgs)
    assert "duplicate-tool-result" in exc_info.value.issues


# ---------------------------------------------------------------------------
# Does not mutate input
# ---------------------------------------------------------------------------


def test_does_not_mutate_input() -> None:
    original = [
        builders.user_message("hi"),
        _assistant_with_tool_call(),
    ]
    original_len = len(original)
    _ = prepare_messages(original)
    assert len(original) == original_len


# ---------------------------------------------------------------------------
# Wiring: stream() and generate() run prepare_messages on input
# ---------------------------------------------------------------------------


async def test_stream_calls_prepare_messages() -> None:
    """stream() should invoke prepare_messages before hitting the adapter."""
    mock_llm([[text_msg("ok")]])
    msgs = [ai.user_message("hi")]

    with patch(
        "ai.models.core.api.integrity.prepare_messages", wraps=lambda m: m
    ) as spy:
        async with models.stream(MOCK_MODEL, msgs) as s:
            async for _ in s:
                pass
        spy.assert_called_once_with(msgs)


async def test_stream_sanitizes_internal_messages() -> None:
    """Internal messages are stripped before reaching the adapter."""
    received: list[list[messages.Message]] = []
    mock = mock_llm([[text_msg("ok")]])

    # Wrap the mock adapter to capture messages it receives
    original_adapter = mock.stream

    async def _spy_stream(
        client: models.Client,
        model: models.Model,
        messages: list[messages.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        received.append(list(messages))
        async for event in original_adapter(
            client, model, messages, tools=tools, output_type=output_type, **kwargs
        ):
            yield event

    models.register_stream("mock", _spy_stream)

    msgs = [
        ai.user_message("hi"),
        messages.Message(role="internal", parts=[messages.TextPart(text="internal")]),
        ai.assistant_message("hello"),
    ]
    async with models.stream(MOCK_MODEL, msgs) as s:
        async for _ in s:
            pass

    # The adapter should have received only 2 messages (internal stripped)
    assert len(received) == 1
    assert len(received[0]) == 2
    assert all(m.role != "internal" for m in received[0])


async def test_generate_calls_prepare_messages() -> None:
    """generate() should invoke prepare_messages before hitting the adapter."""
    sentinel = messages.Message(
        role="assistant",
        parts=[messages.FilePart(data=b"\x89PNG", media_type="image/png")],
    )
    mock_generate([sentinel])
    msgs = [ai.user_message("A cat")]

    with patch(
        "ai.models.core.api.integrity.prepare_messages", wraps=lambda m: m
    ) as spy:
        await models.generate(MOCK_MODEL, msgs, models.ImageParams(n=1))
        spy.assert_called_once_with(msgs)


async def test_generate_sanitizes_internal_messages() -> None:
    """Internal messages are stripped before reaching generate adapter."""
    received: list[list[messages.Message]] = []
    sentinel = messages.Message(
        role="assistant",
        parts=[messages.FilePart(data=b"\x89PNG", media_type="image/png")],
    )

    async def _spy_gen(
        client: models.Client,
        model: models.Model,
        messages: list[messages.Message],
        params: Any,
    ) -> messages.Message:
        received.append(list(messages))
        return sentinel

    models.register_generate("mock", _spy_gen)

    msgs = [
        ai.user_message("A cat"),
        messages.Message(role="internal", parts=[messages.TextPart(text="internal")]),
    ]
    await models.generate(MOCK_MODEL, msgs, models.ImageParams(n=1))

    assert len(received) == 1
    assert len(received[0]) == 1
    assert received[0][0].role == "user"
