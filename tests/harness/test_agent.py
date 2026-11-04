from __future__ import annotations

import asyncio
import pathlib
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest

import ai
from ai import harness
from ai.types import usage


async def _run(
    agent: ai.Agent,
    model: ai.Model,
    messages: list[ai.messages.Message],
    *,
    steer: Sequence[Sequence[str]] = (),
) -> tuple[list[Any], list[ai.messages.Message]]:
    """Run to the end. Answer the n-th steering ask with ``steer[n]``."""
    answers = list(steer)
    got = []
    async with agent.run(model, messages) as stream:
        async for event in stream:
            got.append(event)
            if (
                isinstance(event, ai.events.PendingHookEvent)
                and event.hook.hook_id == harness.STEER
            ):
                texts = answers.pop(0) if answers else []
                ai.resolve_hook(event.hook, harness.Steer(messages=list(texts)))
    assert not answers, "the run asked for steering fewer times than expected"
    return got, stream.messages


def _asks(got: list[Any]) -> int:
    return sum(
        isinstance(e, ai.events.PendingHookEvent)
        and e.hook.hook_id == harness.STEER
        for e in got
    )


@pytest.fixture(autouse=True)
def _cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # keep the real AGENTS.md / skills out of the system prompt
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@ai.tool
async def fast() -> str:
    """Return at once."""
    return "fast done"


async def test_system_prompt_has_env_agents_md_and_skills(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("use uv")
    skill = tmp_path / ".agents/skills/pdf/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: pdf\ndescription: Read PDFs.\n---\nbody")
    model = ai.testing.FakeModel(
        [ai.user_message("hi"), ai.assistant_message("hello")]
    )

    _, messages = await _run(harness.Agent(), model, [ai.user_message("hi")])

    system = model.calls[0][0]
    assert system.role == "system"
    assert f"cwd: {tmp_path}" in system.text
    assert (
        f"Instructions from {tmp_path / 'AGENTS.md'}:\n\nuse uv" in system.text
    )
    assert "<name>pdf</name>" in system.text
    assert "<description>Read PDFs.</description>" in system.text
    assert f"<location>{skill.resolve()}</location>" in system.text
    assert messages[-1].text == "hello"


async def test_caller_system_message_is_kept() -> None:
    model = ai.testing.FakeModel(
        [ai.user_message("hi"), ai.assistant_message("hello")]
    )
    await _run(
        harness.Agent(),
        model,
        [ai.system_message("custom"), ai.user_message("hi")],
    )
    assert [m.text for m in model.calls[0]] == ["custom", "hi"]


async def test_asks_before_each_request_and_before_stopping() -> None:
    call = ai.testing.tool_call(fast)
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message(call),
            ai.assistant_message("done"),
        ]
    )

    class A(harness.Agent):
        TOOLS: ClassVar[list[ai.AgentTool]] = [fast]

    got, _ = await _run(A(), model, [ai.user_message("go")])

    assert _asks(got) == 3
    assert not any(isinstance(e, harness.Steered) for e in got)
    assert not model.unused


async def test_steering_before_the_first_request_keeps_order() -> None:
    model = ai.testing.FakeModel(
        [
            ai.user_message("hi"),
            ai.user_message("a"),
            ai.user_message("b"),
            ai.assistant_message("ok"),
        ]
    )

    got, _ = await _run(
        harness.Agent(), model, [ai.user_message("hi")], steer=[["a", "b"]]
    )

    (steered,) = [e for e in got if isinstance(e, harness.Steered)]
    assert [m.text for m in steered.messages] == ["a", "b"]
    assert not model.unused


async def test_steering_lands_after_the_turns_tools() -> None:
    call = ai.testing.tool_call(fast)
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message(call),
            ai.user_message("also this"),
            ai.assistant_message("ok"),
        ]
    )

    class A(harness.Agent):
        TOOLS: ClassVar[list[ai.AgentTool]] = [fast]

    got, messages = await _run(
        A(), model, [ai.user_message("go")], steer=[[], ["also this"]]
    )

    result = next(e for e in got if isinstance(e, ai.events.ToolCallResult))
    steered = next(e for e in got if isinstance(e, harness.Steered))
    assert got.index(result) < got.index(steered)
    (part,) = [r for m in messages for r in m.tool_results]
    assert part.result == "fast done"
    assert not model.unused


async def test_steering_after_the_final_answer_continues_the_run() -> None:
    model = ai.testing.FakeModel(
        [
            ai.user_message("hi"),
            ai.assistant_message("hello"),
            ai.user_message("again"),
            ai.assistant_message("ok"),
        ]
    )

    _, messages = await _run(
        harness.Agent(), model, [ai.user_message("hi")], steer=[[], ["again"]]
    )

    assert messages[-1].text == "ok"
    assert not model.unused


async def test_max_turns() -> None:
    call = ai.testing.tool_call(fast)
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message(call),
            ai.assistant_message("x"),
        ]
    )

    class A(harness.Agent):
        TOOLS: ClassVar[list[ai.AgentTool]] = [fast]
        MAX_TURNS = 1

    got, _ = await _run(A(), model, [ai.user_message("go")])

    assert isinstance(got[-1], harness.LimitReached)
    assert got[-1].limit == "turns"


async def test_compaction_replaces_old_history_with_summary() -> None:
    first = ai.assistant_message("first").model_copy(
        update={"usage": usage.Usage(input_tokens=1000)}
    )
    summary = ai.user_message(f"{harness.SUMMARY_PREFIX}SUM\n</summary>")
    model = ai.testing.FakeModel(
        [
            ai.user_message("one"),
            first,
            ai.user_message(harness.SUMMARY_PROMPT),
            ai.assistant_message("SUM"),
        ],
        [summary, ai.user_message("two"), ai.assistant_message("done")],
    )

    class A(harness.Agent):
        RESERVE_TOKENS = 10**9  # always over the window
        KEEP_RECENT_TOKENS = 0

    got, messages = await _run(
        A(), model, [ai.user_message("one"), first, ai.user_message("two")]
    )

    compacted = next(e for e in got if isinstance(e, harness.Compacted))
    assert compacted.removed == 2
    assert [m.text for m in messages[1:]] == [summary.text, "two", "done"]
    assert len(model.calls) == 2  # the summary request, then the turn


async def test_task_runs_a_subagent_that_does_not_steer() -> None:
    call = ai.testing.tool_call(harness.task, prompt="look around")
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message(call),
            ai.assistant_message("done"),
        ],
        [ai.user_message("look around"), ai.assistant_message("found it")],
    )

    # a steering ask from the subagent would never be answered here
    got, messages = await asyncio.wait_for(
        _run(harness.Agent(), model, [ai.user_message("go")]), 5
    )

    (result,) = [r for m in messages for r in m.tool_results]
    assert result.model_input == "found it"
    assert messages[-1].text == "done"
    assert _asks(got) == 3
    assert not model.unused
