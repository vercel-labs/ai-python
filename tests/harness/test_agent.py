from __future__ import annotations

import asyncio
import pathlib
import types
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any, ClassVar

import modelsdotdev
import pytest

import ai
from ai import _modelsdev, harness
from ai.types import events as ai_events
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


async def _interrupt(
    agent: ai.Agent,
    model: ai.Model,
    messages: list[ai.messages.Message],
    when: Callable[[Any], bool],
) -> list[ai.messages.Message]:
    """Run until ``when(event)``, then cancel, as the REPL does on Ctrl+C.

    Returns the history the interface keeps (``stream.messages``).
    """
    history = list(messages)

    async def run() -> None:
        async with agent.run(model, history) as stream:
            try:
                async for event in stream:
                    if when(event):
                        task = asyncio.current_task()
                        assert task is not None
                        task.cancel()
            finally:
                history[:] = stream.messages

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.create_task(run()), 5)
    return history


slow_runs: list[str] = []


@ai.tool
async def slow() -> str:
    """Never finish."""
    slow_runs.append("slow")
    await asyncio.sleep(3600)
    return "slow done"


async def test_interrupt_keeps_partial_turn_and_closes_tool_calls() -> None:
    fast_call, slow_call = (
        ai.testing.tool_call(fast),
        ai.testing.tool_call(slow),
    )
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            ai.assistant_message("on it", fast_call, slow_call),
            ai.assistant_message("ok"),
        ]
    )

    class A(harness.Agent):
        TOOLS: ClassVar[list[ai.AgentTool]] = [fast, slow]

    seen: set[str] = set()

    def when_fast_is_done_and_slow_runs(event: Any) -> bool:
        if (
            isinstance(event, ai.events.PendingHookEvent)
            and event.hook.hook_id == harness.STEER
        ):
            ai.resolve_hook(event.hook, harness.Steer())
        elif isinstance(event, ai.events.ToolEnd):
            seen.add(f"end {event.tool_call.tool_name}")
        elif isinstance(event, ai.events.ToolCallResult):
            seen.add("fast result")
        return seen >= {"end slow", "fast result"}

    history = await _interrupt(
        A(), model, [ai.user_message("go")], when_fast_is_done_and_slow_runs
    )

    assistant, tool = history[-2:]
    assert assistant.text == "on it"
    assert [p.tool_name for p in assistant.tool_calls] == ["fast", "slow"]
    results = {r.tool_name: r for r in tool.tool_results}
    assert results["fast"].result == "fast done"
    assert results["slow"].result == harness.INTERRUPTED
    assert results["slow"].is_error

    # the next run goes on from there; it does not run the tools again
    slow_runs.clear()
    _, messages = await _run(A(), model, history)
    assert messages[-1].text == "ok"
    assert not slow_runs
    assert not model.unused


async def test_interrupt_while_waiting_for_steering_keeps_history() -> None:
    model = ai.testing.FakeModel(
        [ai.user_message("go"), ai.assistant_message("x")]
    )

    history = await _interrupt(
        harness.Agent(),
        model,
        [ai.user_message("go")],
        lambda e: isinstance(e, ai.events.PendingHookEvent),
    )

    assert [m.role for m in history] == ["system", "user"]
    assert not model.calls


def test_usd_prices_cache_tiers_and_reasoning() -> None:
    sonnet = [
        modelsdotdev.Cost(input=3, output=15, cache_read=0.3, cache_write=3.75)
    ]
    request = usage.Usage(
        input_tokens=1000,  # 400 of them cache reads
        output_tokens=200,
        cache_read_tokens=400,
        cache_write_tokens=100,  # charged on top
    )
    assert harness.usd(sonnet, request) == pytest.approx(
        (600 * 3 + 400 * 0.3 + 100 * 3.75 + 200 * 15) / 1e6
    )

    tiers = [
        modelsdotdev.Cost(input=1, output=0),
        modelsdotdev.Cost(input=2, output=0, min_context=200_000),
    ]
    assert harness.usd(tiers, usage.Usage(input_tokens=1_000_000)) == 2

    priced = [modelsdotdev.Cost(input=0, output=10, reasoning=1)]
    request = usage.Usage(output_tokens=1_000_000, reasoning_tokens=400_000)
    assert harness.usd(priced, request) == pytest.approx(6 + 0.4)


def _priced(monkeypatch: pytest.MonkeyPatch) -> None:
    """$1 per million input tokens for every model; FakeModel reports the
    ``usage`` of each scripted message."""
    info = types.SimpleNamespace(
        cost=[modelsdotdev.Cost(input=1, output=0)],
        limits=types.SimpleNamespace(context=10**9),  # no compaction
    )
    monkeypatch.setattr(_modelsdev, "get_model_by_id", lambda _id: info)

    def stream(
        self: Any, model: Any, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[ai.events.Event]:
        response = self._next_response(messages)

        async def events() -> AsyncGenerator[ai.events.Event]:
            async for event in ai_events._replay_message_events(response):
                if isinstance(event, ai.events.StreamEnd):
                    event = event.model_copy(update={"usage": response.usage})
                yield event

        return events()

    monkeypatch.setattr(ai.testing._FakeProvider, "stream", stream)


def _costs(dollars: float, *parts: Any) -> ai.messages.Message:
    """An assistant message whose request cost *dollars* (see `_priced`)."""
    return ai.assistant_message(*parts).model_copy(
        update={"usage": usage.Usage(input_tokens=int(dollars * 1e6))}
    )


def _leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for e in exc.exceptions for leaf in _leaves(e)]
    return [exc]


async def test_max_usd_without_prices_errors_before_the_first_request() -> None:
    model = ai.testing.FakeModel(
        [ai.user_message("go"), ai.assistant_message("x")]
    )

    class A(harness.Agent):
        MAX_USD = 1.0

    with pytest.raises(BaseException) as info:
        await _run(A(), model, [ai.user_message("go")])

    (error,) = _leaves(info.value)
    assert isinstance(error, harness.UnknownCostError)
    assert "fake-model" in str(error)
    assert not model.calls


async def test_max_usd_errors_when_usage_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _priced(monkeypatch)
    model = ai.testing.FakeModel(
        [ai.user_message("go"), ai.assistant_message("no usage")]
    )

    class A(harness.Agent):
        MAX_USD = 1.0

    with pytest.raises(BaseException) as info:
        await _run(A(), model, [ai.user_message("go")])

    (error,) = _leaves(info.value)
    assert isinstance(error, harness.UnknownCostError)


async def test_subagent_spend_counts_against_the_parent_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _priced(monkeypatch)
    call = ai.testing.tool_call(harness.task, prompt="look around")
    model = ai.testing.FakeModel(
        [
            ai.user_message("go"),
            _costs(0.5, call),
            ai.assistant_message("done"),
        ],
        [ai.user_message("look around"), _costs(1.0, "found it")],
    )

    class A(harness.Agent):
        MAX_USD = 1.5  # the parent alone spends only 0.5

    got, _ = await _run(A(), model, [ai.user_message("go")])

    assert isinstance(got[-1], harness.LimitReached)
    assert got[-1].limit == "usd"
    assert got[-1].usd == pytest.approx(1.5)
    assert got[-1].usage.input_tokens == 1_500_000
    assert [m.text for m in model.unused] == ["done"]


async def test_subagent_stops_at_the_shared_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _priced(monkeypatch)
    call = ai.testing.tool_call(harness.task, prompt="look around")
    model = ai.testing.FakeModel(
        [ai.user_message("go"), _costs(1.0, call), ai.assistant_message("x")],
        [ai.user_message("look around"), _costs(1.0, "found it")],
    )

    class A(harness.Agent):
        MAX_USD = 1.0  # spent by the parent's first turn

    got, _ = await asyncio.wait_for(
        _run(A(), model, [ai.user_message("go")]), 5
    )

    assert isinstance(got[-1], harness.LimitReached)
    assert len(model.calls) == 1  # neither the subagent nor the parent asked
