"""The session state machine, offline: ordering, stop, abandoned turns.

Driven by `fakes.FakeAdapter`, so every case runs in milliseconds and
without a CLI. The e2e suite proves the same contract against real agents.
"""

from __future__ import annotations

import asyncio
import contextvars
import warnings
from typing import Any

import pytest

from ai.harnesses.experimental import _approval, _session, errors
from ai.types import events
from ai.types import messages as m

from . import fakes


def roles(messages: list[m.Message]) -> list[str]:
    return [message.role for message in messages]


def kinds(message: m.Message) -> list[str]:
    return [part.kind for part in message.parts]


# -- plain turns ---------------------------------------------------------------


async def test_a_plain_turn_settles_with_its_text() -> None:
    harness, _ = await fakes.open_harness()
    result = await harness.session().run("hello")
    assert result.text == "hello"
    assert result.finish_reason == "stop"
    assert roles(result.messages) == ["user", "assistant"]


async def test_turns_accumulate_in_one_session() -> None:
    harness, _ = await fakes.open_harness()
    session = harness.session()
    await session.run("one")
    result = await session.run("two")
    assert roles(result.messages) == ["user", "assistant", "user", "assistant"]
    assert result.text == "two"


# -- H1: message order -------------------------------------------------------


def _tool_turn(prompt: str) -> list[fakes.Step]:
    return [
        events.StreamStart(),
        *fakes.text("I'll run it.", block="a"),
        *fakes.tool("c1"),
        *fakes.text("done", block="b"),
    ]


async def test_a_tool_call_comes_before_its_result() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))
    result = await harness.session().run("go")
    assert roles(result.messages) == ["user", "assistant", "tool", "assistant"]
    first, tool, last = result.messages[1:]
    assert kinds(first) == ["text", "tool_call"]
    assert isinstance(first.parts[1], m.ToolCallPart)
    assert first.parts[1].tool_args == "{}"
    assert isinstance(tool.parts[0], m.ToolResultPart)
    assert tool.parts[0].tool_call_id == "c1"
    assert kinds(last) == ["text"]
    assert result.text == "done", "the answer, not the narration"


async def test_two_tool_rounds_alternate() -> None:
    def script(prompt: str) -> list[fakes.Step]:
        return [
            events.StreamStart(),
            *fakes.tool("c1"),
            *fakes.tool("c2"),
            *fakes.text("done"),
        ]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    result = await harness.session().run("go")
    assert roles(result.messages) == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    ids = [message.id for message in result.messages]
    assert len(set(ids)) == len(ids)


async def test_parallel_calls_share_one_assistant_message() -> None:
    def script(prompt: str) -> list[fakes.Step]:
        one, two = fakes.tool("c1"), fakes.tool("c2")
        # Both calls first, then both results — how claude reports a turn
        # that asked for two tools at once.
        return [events.StreamStart(), *one[:3], *two[:3], one[3], two[3]]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    result = await harness.session().run("go")
    assert roles(result.messages) == ["user", "assistant", "tool", "tool"]
    assert kinds(result.messages[1]) == ["tool_call", "tool_call"]


async def test_history_mid_turn_already_has_the_call() -> None:
    """What an approval hook sees: the call it is asked about is in history."""
    seen: list[list[str]] = []
    harness: Any = None

    def script(prompt: str) -> list[fakes.Step]:
        async def look() -> None:
            sid = next(iter(harness._sessions))
            seen.append(roles(harness._history_of(sid)))

        steps = fakes.tool("c1")
        return [events.StreamStart(), *steps[:3], fakes.Run(look), steps[3]]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    await harness.session().run("go")
    assert seen == [["user", "assistant"]]


async def test_a_turn_stopped_mid_tool_has_no_orphan_result() -> None:
    pause = fakes.Pause()

    def script(prompt: str) -> list[fakes.Step]:
        steps = fakes.tool("c1")
        return [events.StreamStart(), *steps[:3], pause, steps[3]]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    session = harness.session()
    turn = session.stream("go")
    async for event in turn:
        if isinstance(event, events.ToolEnd):
            await session.stop()
    assert turn.result.cancelled
    assert roles(turn.result.messages) == ["user", "assistant"]
    assert kinds(turn.result.messages[1]) == ["tool_call"]


async def test_a_steer_lands_where_it_was_sent() -> None:
    session: Any = None

    async def steer() -> None:
        await session.steer("smaller, please")

    def script(prompt: str) -> list[fakes.Step]:
        return [
            events.StreamStart(),
            *fakes.text("before", block="a"),
            fakes.Run(steer),
            *fakes.text("after", block="b"),
        ]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    session = harness.session()
    result = await session.run("go")
    assert roles(result.messages) == ["user", "assistant", "user", "assistant"]
    steered = result.messages[2].parts[0]
    assert isinstance(steered, m.TextPart)
    assert steered.text == "smaller, please"


async def test_the_answer_is_the_adapters_final_message() -> None:
    """Codex streams its preamble but leaves it out of the final message;
    `text` follows the adapter, history keeps what was streamed."""

    def script(prompt: str) -> list[fakes.Step]:
        return [
            events.StreamStart(),
            *fakes.text("Let me think. ", block="a"),
            *fakes.text("42", block="b"),
            events.StreamEnd(
                message=m.Message(
                    role="assistant", parts=[m.TextPart(text="42")]
                ),
                finish_reason="stop",
            ),
        ]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    result = await harness.session().run("go")
    assert result.text == "42"
    assert kinds(result.messages[1]) == ["text", "text"]


async def test_an_unstreamed_final_message_still_lands_in_history() -> None:
    def script(prompt: str) -> list[fakes.Step]:
        return [
            events.StreamStart(),
            events.StreamEnd(
                message=m.Message(
                    role="assistant", parts=[m.TextPart(text="whole")]
                ),
                finish_reason="stop",
            ),
        ]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    result = await harness.session().run("go")
    assert result.text == "whole"
    assert roles(result.messages) == ["user", "assistant"]


async def test_provider_metadata_of_the_final_message_is_kept() -> None:
    def script(prompt: str) -> list[fakes.Step]:
        return [
            events.StreamStart(),
            *fakes.text("hi"),
            events.StreamEnd(
                message=m.Message(
                    role="assistant",
                    parts=[m.TextPart(text="hi")],
                    provider_metadata={"x": {"y": 1}},
                ),
                finish_reason="stop",
            ),
        ]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    result = await harness.session().run("go")
    assert result.messages[-1].provider_metadata == {"x": {"y": 1}}


# -- H2: stop from anywhere ---------------------------------------------------


def _paused(pause: fakes.Pause) -> Any:
    def script(prompt: str) -> list[fakes.Step]:
        return [events.StreamStart(), *fakes.text("partial"), pause]

    return script


async def test_stop_from_another_task_settles_the_turn() -> None:
    pause = fakes.Pause()
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(_paused(pause))
    )
    session = harness.session()
    turn = session.stream("go")

    async def consume() -> None:
        async for _ in turn:
            pass

    task = asyncio.create_task(consume())
    await pause.reached.wait()
    await session.stop()
    assert turn.settled, "stop returns once the turn has settled"
    assert turn.result.cancelled
    assert turn.result.text == "partial"
    await asyncio.wait_for(task, 1)
    assert adapter.stops == [session.session_id]


async def test_stop_inside_the_loop_body_still_works() -> None:
    pause = fakes.Pause()
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(_paused(pause))
    )
    session = harness.session()
    turn = session.stream("go")
    async for event in turn:
        if isinstance(event, events.TextDelta):
            await session.stop()
    assert turn.result.cancelled
    assert adapter.stops == [session.session_id]


async def test_stop_without_a_turn_does_nothing() -> None:
    harness, adapter = await fakes.open_harness()
    session = harness.session()
    await session.stop()
    await session.run("hi")
    await session.stop()
    assert adapter.stops == []


async def test_stop_gives_up_on_an_adapter_that_never_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_session, "STOP_TIMEOUT", 0.2)
    pause = fakes.Pause()
    harness, _ = await fakes.open_harness(
        fakes.FakeAdapter(_paused(pause), honors_stop=False)
    )
    session = harness.session()
    turn = session.stream("go")

    async def consume() -> None:
        async for _ in turn:
            pass

    task = asyncio.create_task(consume())
    await pause.reached.wait()
    await asyncio.wait_for(session.stop(), 1)
    pause.release.set()
    await asyncio.wait_for(task, 1)
    assert turn.result.cancelled, "a stop asked for is a cancellation"


async def test_a_second_run_waits_for_a_turn_running_in_another_task() -> None:
    pause = fakes.Pause()
    harness, _ = await fakes.open_harness(
        fakes.FakeAdapter(
            lambda prompt: (
                _paused(pause)(prompt)
                if prompt == "slow"
                else fakes.reply(prompt)
            )
        )
    )
    session = harness.session()
    slow = asyncio.create_task(session.run("slow"))
    await pause.reached.wait()
    fast = asyncio.create_task(session.run("fast"))
    await asyncio.sleep(0.05)
    assert not fast.done(), "one turn at a time"
    pause.release.set()
    assert (await slow).text == "partial"
    assert (await fast).text == "fast"


# -- H5: walking away from a turn ---------------------------------------------


async def test_breaking_out_stops_the_agent_before_the_next_turn() -> None:
    pause = fakes.Pause()
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(
            lambda prompt: (
                _paused(pause)(prompt)
                if prompt == "long"
                else fakes.reply(prompt)
            )
        )
    )
    session = harness.session()
    async for event in session.stream("long"):
        if isinstance(event, events.TextDelta):
            break
    result = await session.run("next")
    assert adapter.stops == [session.session_id], "the CLI's turn was ended"
    assert "long" in adapter.turns_finalized, "the adapter's turn was closed"
    assert result.text == "next"
    assert roles(result.messages) == ["user", "assistant", "user", "assistant"]


async def test_a_deadline_stops_once_and_settles_cancelled() -> None:
    pause = fakes.Pause()
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(_paused(pause), honors_stop=False)
    )
    session = harness.session()
    result = await session.run("go", timeout=0.1)
    assert result.cancelled
    assert result.text == "partial"
    assert adapter.stops == [session.session_id]


async def test_external_cancellation_propagates_and_frees_the_session() -> None:
    pause = fakes.Pause()
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(
            lambda prompt: (
                _paused(pause)(prompt)
                if prompt == "slow"
                else fakes.reply(prompt)
            ),
            honors_stop=False,
        )
    )
    session = harness.session()
    task = asyncio.create_task(session.run("slow"))
    await pause.reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert adapter.stops == [session.session_id]
    assert (await session.run("after")).text == "after"


# -- H7: per-session state ---------------------------------------------------


async def test_approval_errors_belong_to_the_session_they_happened_in() -> None:
    pause = fakes.Pause()
    adapter: Any = None

    def script(prompt: str) -> list[fakes.Step]:
        if prompt != "a":
            return fakes.reply(prompt)

        async def fail() -> None:
            adapter.on_approval_error(a.session_id, "hook raised")

        return [events.StreamStart(), fakes.Run(fail), pause, *fakes.text("a")]

    harness, adapter = await fakes.open_harness(fakes.FakeAdapter(script))
    a, b = harness.session(), harness.session()
    first = asyncio.create_task(a.run("a"))
    await pause.reached.wait()
    other = await b.run("b")
    pause.release.set()
    mine = await first
    assert other.approval_errors == []
    assert mine.approval_errors == ["hook raised"]


async def test_a_closed_session_is_forgotten_by_its_harness() -> None:
    harness, _ = await fakes.open_harness()
    session = harness.session()
    await session.run("hi")
    assert session.session_id in harness._sessions
    await session.close()
    assert session.session_id not in harness._sessions
    await harness.run("throwaway")
    assert harness._sessions == {}


# -- H8: a steer after the agent finished -------------------------------------


async def test_a_steer_after_the_turn_ended_is_refused() -> None:
    """Claude would take it as a new turn of its own, and every later turn
    would read the answer of the one before (measured)."""
    harness, adapter = await fakes.open_harness()
    session = harness.session()
    turn = session.stream("hi")
    async for event in turn:
        if isinstance(event, events.StreamEnd):
            with pytest.raises(RuntimeError, match="no active turn"):
                await session.steer("too late")
    assert adapter.steers == []
    assert roles(turn.result.messages) == ["user", "assistant"]


# -- H11: which CLI version ---------------------------------------------------


@pytest.mark.parametrize(
    ("version", "warns"),
    [("fake 1.9.3", True), ("fake 2.0.0", False), ("fake 2.1.7", False)],
)
async def test_a_cli_older_than_the_verified_one_warns(
    version: str, warns: bool
) -> None:
    adapter = fakes.FakeAdapter()
    adapter.version = version
    adapter.verified_version = "2.0.0"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await fakes.open_harness(adapter)
    older = [w for w in caught if "older than" in str(w.message)]
    assert bool(older) is warns


async def test_an_unreadable_version_does_not_warn() -> None:
    adapter = fakes.FakeAdapter()
    adapter.version = "unknown build"
    adapter.verified_version = "2.0.0"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await fakes.open_harness(adapter)


async def test_a_failed_turn_says_which_cli_version_failed() -> None:
    def script(prompt: str) -> list[fakes.Step]:
        async def fail() -> None:
            raise errors.TurnFailedError("fake", "boom")

        return [events.StreamStart(), fakes.Run(fail)]

    harness, _ = await fakes.open_harness(fakes.FakeAdapter(script))
    with pytest.raises(errors.TurnFailedError) as info:
        await harness.session().run("go")
    assert "fake 1.0" in "\n".join(getattr(info.value, "__notes__", []))


# -- the approval hook runs inside its turn ----------------------------------

_who: contextvars.ContextVar[str] = contextvars.ContextVar("who", default="?")


def _asking(adapter: fakes.AskingAdapter) -> Any:
    def script(prompt: str) -> list[fakes.Step]:
        return [events.StreamStart(), adapter.ask(), *fakes.text(prompt)]

    return script


async def test_the_approval_hook_sees_its_turns_context() -> None:
    """Not the context of whoever started the CLI client's reader."""
    seen: list[str] = []

    async def approve(ctx: _approval.ApprovalContext) -> _approval.Decision:
        seen.append(_who.get())
        return _approval.Allow()

    adapter = fakes.AskingAdapter()
    adapter.script = _asking(adapter)
    _who.set("open")
    harness, _ = await fakes.open_harness(adapter, approve=approve)
    session = harness.session()
    _who.set("turn 1")
    await session.run("one")
    _who.set("turn 2")
    await session.run("two")
    assert seen == ["turn 1", "turn 2"]
    assert adapter.decisions == [_approval.Allow(), _approval.Allow()]


async def test_concurrent_approvals_do_not_share_contextvar_writes() -> None:
    seen: list[str] = []
    both = asyncio.Barrier(2)

    async def approve(ctx: _approval.ApprovalContext) -> _approval.Decision:
        _who.set(ctx.call.id)
        await both.wait()
        seen.append(_who.get())
        return _approval.Allow()

    adapter = fakes.AskingAdapter()

    def script(prompt: str) -> list[fakes.Step]:
        first, second = adapter.ask(call_id="a"), adapter.ask(call_id="b")

        async def ask_both() -> None:
            await asyncio.gather(first.fn(), second.fn())

        return [events.StreamStart(), fakes.Run(ask_both), *fakes.text("ok")]

    adapter.script = script
    harness, _ = await fakes.open_harness(adapter, approve=approve)
    await harness.session().run("go")
    assert sorted(seen) == ["a", "b"]


async def test_no_hook_installs_no_gate() -> None:
    _, adapter = await fakes.open_harness()
    assert adapter.approve is None


# -- the turn reports the approval it waits on --------------------------------


async def test_a_turn_reports_the_approval_it_is_blocked_on() -> None:
    release = asyncio.Event()

    async def approve(ctx: _approval.ApprovalContext) -> _approval.Decision:
        await release.wait()
        return _approval.Deny("not that")

    adapter = fakes.AskingAdapter()
    adapter.script = _asking(adapter)
    harness, _ = await fakes.open_harness(adapter, approve=approve)
    seen: list[str] = []
    async with harness.session().stream("go") as turn:
        async for event in turn:
            seen.append(type(event).__name__)
            if isinstance(event, events.RunBlocked):
                # Delivered while the hook is still deciding: the CLI's own
                # stream is silent meanwhile.
                assert turn.blocked
                assert [h.tool_call_id for h in event.hooks] == ["c1"]
                assert [h.hook_id for h in turn.deferred_hooks] == [
                    "approve_c1"
                ]
                release.set()
            elif isinstance(event, events.HookEvent):
                assert event.hook.hook_type == _session.APPROVAL_HOOK_TYPE
                assert event.hook.metadata == {"tool": "Bash", "kwargs": {}}
                if event.hook.status == "resolved":
                    assert event.hook.resolution is not None
                    assert event.hook.resolution["decision"] == "deny"
                    assert not turn.blocked
    assert seen[:4] == [
        "StreamStart",
        "PendingHookEvent",
        "RunBlocked",
        "ResolvedHookEvent",
    ]
    assert not turn.blocked
    assert turn.output == "go"
    assert roles(turn.messages) == [
        "user",
        "assistant",
    ], "approval notices are not part of the transcript"


async def test_a_failed_approval_is_reported_cancelled() -> None:
    async def approve(ctx: _approval.ApprovalContext) -> _approval.Decision:
        raise RuntimeError("policy service down")

    adapter = fakes.AskingAdapter()
    adapter.script = _asking(adapter)
    harness, _ = await fakes.open_harness(adapter, approve=approve)
    turn = harness.session().stream("go")
    hooks = [e.hook async for e in turn if isinstance(e, events.HookEvent)]
    assert [h.status for h in hooks] == ["pending", "cancelled"]
    assert not turn.blocked


# -- the turn, shaped like `AgentStream` --------------------------------------


async def test_a_turn_is_a_context_manager_that_settles_on_exit() -> None:
    pause = fakes.Pause()
    harness, adapter = await fakes.open_harness(
        fakes.FakeAdapter(_paused(pause))
    )
    session = harness.session()
    async with session.stream("go") as turn:
        async for event in turn:
            if isinstance(event, events.TextDelta):
                assert roles(turn.messages) == ["user", "assistant"], "live"
                break
    assert turn.settled
    assert turn.result.cancelled
    assert adapter.stops == [session.session_id]


async def test_output_defaults_to_the_final_text() -> None:
    harness, _ = await fakes.open_harness(fakes.FakeAdapter(_tool_turn))
    async with harness.session().stream("go") as turn:
        async for _ in turn:
            pass
    assert turn.output == turn.result.text == "done"
    assert turn.usage == turn.result.usage


async def test_output_is_unavailable_until_the_turn_settles() -> None:
    harness, _ = await fakes.open_harness()
    turn = harness.session().stream("go")
    with pytest.raises(RuntimeError, match="not settled"):
        _ = turn.output


@pytest.mark.parametrize("bad", ["/etc", "../x", "a/b", "-rf", "", ".."])
async def test_every_harness_refuses_a_session_id_that_is_a_path(
    bad: str,
) -> None:
    """Security review: ids become lock directories, transcript paths and
    CLI arguments in the adapters. Refused where they enter, on every
    harness, before an adapter sees one."""
    from ai.harnesses.experimental.errors import ResumeFailedError

    harness, adapter = await fakes.open_harness()
    calls = [
        harness.resume(bad),
        harness.fork(bad),
        harness.history(bad),
        harness.tui(bad),
    ]
    for call in calls:
        with pytest.raises(ResumeFailedError, match="not a session id"):
            await call
    assert adapter.turns_started == []
