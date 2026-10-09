"""What peek draws, from the SDK's own message types: no sandbox needed."""

from __future__ import annotations

import json

from afk.render import Style, lines
from afk.rows import CONTINUE, CONTINUE_MARK, turn_finished
from afk.verbs import _peek_start  # noqa: PLC2701

from ai.types.messages import Message, TextPart, ToolCallPart, ToolResultPart

PLAIN = Style(color=False)


def _call(name: str, **args: object) -> ToolCallPart:
    return ToolCallPart(
        tool_call_id="c1", tool_name=name, tool_args=json.dumps(args)
    )


def _result(value: object, *, error: bool = False) -> ToolResultPart:
    return ToolResultPart(
        tool_call_id="c1",
        tool_name="Bash",
        result=value,
        result_kind="error" if error else "json",
    )


def test_user_words_are_marked_and_reminders_are_not_shown() -> None:
    m = Message(
        role="user",
        parts=[
            TextPart(
                text=(
                    "<system-reminder>internal</system-reminder>"
                    "fix the add() bug"
                )
            )
        ],
    )
    assert lines(m, 80, PLAIN) == ["› fix the add() bug"]  # noqa: RUF001


def test_a_message_of_only_reminders_draws_nothing() -> None:
    assert (
        lines(
            Message(
                role="user",
                parts=[TextPart(text="<system-reminder>x</system-reminder>")],
            ),
            80,
            PLAIN,
        )
        == []
    )


def test_agent_text_wraps_to_the_terminal_and_says_when_it_is_cut() -> None:
    m = Message(role="assistant", parts=[TextPart(text="word " * 400)])
    drawn = lines(m, 40, PLAIN)
    assert all(len(ln) <= 40 for ln in drawn)
    assert drawn[-1].startswith("  … ") and drawn[-1].endswith("more lines")
    assert len(drawn) == 12 + 1


def test_a_tool_call_is_one_line_naming_what_it_does() -> None:
    m = Message(
        role="assistant",
        parts=[
            _call(
                "Bash", command="python3  test_calc.py", description="run tests"
            )
        ],
    )
    assert lines(m, 80, PLAIN) == ["  ⏺ Bash python3 test_calc.py"]
    m = Message(
        role="assistant",
        parts=[
            _call(
                "Edit", file_path="/vercel/sandbox/calc.py", old_string="a - b"
            )
        ],
    )
    assert lines(m, 80, PLAIN) == ["  ⏺ Edit /vercel/sandbox/calc.py"]


def test_a_tool_result_is_its_first_line_and_a_count() -> None:
    m = Message(role="tool", parts=[_result("3 passed\nline two\nline three")])
    assert lines(m, 80, PLAIN) == ["    ⎿ 3 passed  (+2 lines)"]
    m = Message(role="tool", parts=[_result("Traceback: boom", error=True)])
    assert lines(m, 80, PLAIN) == ["    ⎿ error: Traceback: boom"]
    m = Message(role="tool", parts=[_result("")])
    assert lines(m, 80, PLAIN) == ["    ⎿ done"]


def test_long_lines_are_clipped_to_the_width() -> None:
    m = Message(role="assistant", parts=[_call("Bash", command="x" * 200)])
    (line,) = lines(m, 50, PLAIN)
    assert len(line) == 50 and line.endswith("…")


def test_a_turn_is_finished_only_on_the_agents_own_words() -> None:
    said = Message(role="assistant", parts=[TextPart(text="Fixed and tested.")])
    calling = Message(
        role="assistant",
        parts=[TextPart(text="Running it."), _call("Bash", command="ls")],
    )
    result = Message(role="tool", parts=[_result("ok")])
    asked = Message(role="user", parts=[TextPart(text="go")])
    assert turn_finished([asked, said])
    assert not turn_finished(
        [asked, calling]
    ), "a pending tool call means the turn goes on"
    assert not turn_finished(
        [asked, calling, result]
    ), "a result means the agent speaks next"
    assert not turn_finished([asked]) and not turn_finished([])


def test_bold_markers_are_not_drawn_as_asterisks() -> None:
    m = Message(role="assistant", parts=[TextPart(text="**add()** now adds")])
    assert lines(m, 80, PLAIN) == ["  add() now adds"]


def test_peek_starts_at_the_prompt_of_the_current_turn() -> None:
    ask = Message(role="user", parts=[TextPart(text="go")])
    call = Message(role="assistant", parts=[_call("Bash", command="ls")])
    res = Message(role="tool", parts=[_result("ok")])
    done = Message(role="assistant", parts=[TextPart(text="done")])
    history = [ask, done, ask, call, res, call, res, call, res, done]
    assert _peek_start(history) == 2, "the turn reads whole, from its prompt"
    no_prompt = [call, res, call, res, call, res, done]
    start = _peek_start(no_prompt)
    assert (
        no_prompt[start].role != "tool"
    ), "never start on a result whose call is not shown"


def test_a_pushed_turn_is_not_finished_before_it_shows_up() -> None:
    past = [
        Message(role="user", parts=[TextPart(text="plan it")]),
        Message(role="assistant", parts=[TextPart(text="Ready.")]),
    ]
    assert not turn_finished(
        past, CONTINUE_MARK
    ), "the old last reply proves nothing about the new turn"
    asked = [
        *past,
        Message(
            role="user",
            parts=[TextPart(text=CONTINUE.format(path="/vercel/sandbox"))],
        ),
    ]
    assert not turn_finished(asked, CONTINUE_MARK)
    working = [
        *asked,
        Message(role="assistant", parts=[_call("Bash", command="ls")]),
    ]
    assert not turn_finished(working, CONTINUE_MARK)
    done = [
        *working,
        Message(role="tool", parts=[_result("ok")]),
        Message(role="assistant", parts=[TextPart(text="Done.")]),
    ]
    assert turn_finished(done, CONTINUE_MARK)
