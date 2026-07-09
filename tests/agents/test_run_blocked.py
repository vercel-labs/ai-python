"""RunBlocked: the run-is-blocked-on-hooks signal.

A run is blocked when at least one hook is pending, no model stream
is producing events, and every in-flight tool call is suspended
awaiting a hook.  ``RunStateTracker`` folds the event stream and a
``RunBlocked`` event is emitted when the run blocks; there is no
mirror event — a blocked run can only resume via a hook resolution,
so the non-``pending`` ``HookEvent`` is the unblock signal.
``AgentStream`` folds both into its ``blocked`` / ``pending_hooks``
properties.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pydantic

import ai
from ai.types import events as events_
from ai.types import messages as messages_

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg


class Confirmation(pydantic.BaseModel):
    approved: bool


@ai.tool(require_approval=True)
async def gated(x: int) -> str:
    """A tool that requires approval."""
    return f"gated ran with {x}"


def _approve(hook: messages_.HookPart[Any]) -> None:
    ai.resolve_hook(
        hook.hook_id, ai.tools.ToolApproval(granted=True, reason="ok")
    )


def _multi_call_msg(*tc: tuple[str, str]) -> messages_.Message:
    """Assistant message with several (tool_call_id, tool_name) calls."""
    return messages_.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id=tc_id, tool_name=name, tool_args='{"x": 1}'
            )
            for tc_id, name in tc
        ],
    )


async def test_gated_tool_block_cycle() -> None:
    """Park on approval -> RunBlocked; the resolved HookEvent unblocks."""
    my_agent = ai.Agent(tools=[gated])
    mock_llm(
        [
            [tool_call_msg(name="gated", args='{"x": 1}')],
            [text_msg("done", id="msg-2")],
        ]
    )

    delivered: list[events_.AgentEvent] = []

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            delivered.append(event)
            if isinstance(event, events_.RunBlocked):
                assert stream.blocked
                assert [h.hook_id for h in stream.pending_hooks] == [
                    h.hook_id for h in event.hooks
                ]
                assert event.hooks[0].tool_call_id == "tc-1"
                _approve(event.hooks[0])
            elif (
                isinstance(event, events_.HookEvent)
                and event.hook.status == "resolved"
            ):
                # The resolution is the unblock signal; the stream's
                # fold has already applied it when the event arrives.
                assert not stream.blocked

    assert not stream.blocked
    assert stream.pending_hooks == []

    # Ordering: the run blocks after the hook parks (and only once
    # the model stream stops producing).
    def index(pred: Any) -> int:
        [idx] = [i for i, e in enumerate(delivered) if pred(e)]
        return idx

    pending_idx = index(
        lambda e: isinstance(e, events_.HookEvent)
        and e.hook.status == "pending"
    )
    resolved_idx = index(
        lambda e: isinstance(e, events_.HookEvent)
        and e.hook.status == "resolved"
    )
    blocked_idx = index(lambda e: isinstance(e, events_.RunBlocked))
    assert pending_idx < blocked_idx < resolved_idx

    assert stream.output == "done"


async def test_busy_tool_defers_block_signal() -> None:
    """No block signal while an unblocked tool is still running."""
    release = asyncio.Event()

    @ai.tool
    async def slow(x: int) -> str:
        """A slow tool."""
        await release.wait()
        return "slow done"

    my_agent = ai.Agent(tools=[gated, slow])
    mock_llm(
        [
            [_multi_call_msg(("tc-1", "gated"), ("tc-2", "slow"))],
            [text_msg("done", id="msg-2")],
        ]
    )

    delivered: list[events_.AgentEvent] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            delivered.append(event)
            if (
                isinstance(event, events_.HookEvent)
                and event.hook.status == "pending"
            ):
                assert not stream.blocked
                release.set()
            if isinstance(event, events_.RunBlocked):
                _approve(event.hooks[0])

    [slow_idx] = [
        i
        for i, e in enumerate(delivered)
        if isinstance(e, events_.ToolCallResult)
        and e.results[0].tool_call_id == "tc-2"
    ]
    [blocked_idx] = [
        i for i, e in enumerate(delivered) if isinstance(e, events_.RunBlocked)
    ]
    assert blocked_idx > slow_idx


async def test_parallel_gated_tools() -> None:
    """Blocking needs *every* in-flight tool parked, and the signal
    re-fires when the run parks again on the remaining hook."""
    my_agent = ai.Agent(tools=[gated])
    mock_llm(
        [
            [_multi_call_msg(("tc-1", "gated"), ("tc-2", "gated"))],
            [text_msg("done", id="msg-2")],
        ]
    )

    hook_counts: list[int] = []

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if isinstance(event, events_.RunBlocked):
                hook_counts.append(len(event.hooks))
                _approve(event.hooks[0])

    # Blocks on both hooks, resumes, then re-blocks on the remaining one.
    assert hook_counts == [2, 1]
    assert not stream.blocked


async def test_loop_level_hook() -> None:
    """A hook awaited directly by the loop (no tool task) blocks too."""

    class MyAgent(ai.Agent):
        async def loop(
            self, context: ai.Context
        ) -> AsyncGenerator[events_.AgentEvent]:
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            await ai.hook("confirm", payload=Confirmation)

    mock_llm([[text_msg("OK")]])

    blocked_events: list[events_.RunBlocked] = []
    async with MyAgent().run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if isinstance(event, events_.RunBlocked):
                blocked_events.append(event)
                assert event.hooks[0].hook_id == "confirm"
                ai.resolve_hook("confirm", {"approved": True})

    assert len(blocked_events) == 1
    assert not stream.blocked


async def test_abort_leaves_blocked() -> None:
    """Serverless abort: the run ends still blocked, hooks still pending."""
    my_agent = ai.Agent(tools=[gated])
    mock_llm([[tool_call_msg(name="gated", args='{"x": 1}')]])

    blocked_events: list[events_.RunBlocked] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            if isinstance(event, events_.RunBlocked):
                blocked_events.append(event)
            if (
                isinstance(event, events_.HookEvent)
                and event.hook.status == "pending"
            ):
                ai.abort_pending_hook(event.hook)

    # The abort settles the tool with an is_hook_pending placeholder,
    # but no hook resolution ever arrives: the run ends still blocked.
    assert len(blocked_events) == 1
    assert stream.blocked
    assert len(stream.pending_hooks) == 1
    assert stream.messages[-1].tool_results[0].is_hook_pending


async def test_no_hooks_no_block_events() -> None:
    @ai.tool
    async def plain(x: int) -> str:
        """A plain tool."""
        return "plain done"

    my_agent = ai.Agent(tools=[plain])
    mock_llm(
        [
            [tool_call_msg(name="plain", args='{"x": 1}')],
            [text_msg("done", id="msg-2")],
        ]
    )

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            assert not isinstance(event, events_.RunBlocked)

    assert not stream.blocked
    assert stream.pending_hooks == []


async def test_block_signal_waits_for_stream_end() -> None:
    """A park while the model is still streaming is not a blocked run:
    the signal is deferred until the stream stops producing."""
    msg = messages_.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id="tc-1", tool_name="gated", tool_args='{"x": 1}'
            ),
            messages_.TextPart(text="still streaming..."),
        ],
    )
    my_agent = ai.Agent(tools=[gated])
    mock_llm([[msg], [text_msg("done", id="msg-2")]])

    delivered: list[events_.AgentEvent] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            delivered.append(event)
            if isinstance(event, events_.RunBlocked):
                _approve(event.hooks[0])

    blocked = [
        i for i, e in enumerate(delivered) if isinstance(e, events_.RunBlocked)
    ]
    stream_ends = [
        i for i, e in enumerate(delivered) if isinstance(e, events_.StreamEnd)
    ]
    assert len(blocked) == 1
    assert blocked[0] > stream_ends[0]


async def test_unattributed_hook_in_tool_fails_closed() -> None:
    """A hook awaited inside a tool without ``tool_call_id=`` cannot be
    attributed, so the in-flight tool reads as busy and no block signal
    is emitted (fail closed, never a false "waiting for you")."""

    @ai.tool
    async def self_gated(x: int) -> str:
        """Parks on its own unattributed hook."""
        await ai.hook("self_gate", payload=Confirmation)
        return "ran"

    my_agent = ai.Agent(tools=[self_gated])
    mock_llm(
        [
            [tool_call_msg(name="self_gated", args='{"x": 1}')],
            [text_msg("done", id="msg-2")],
        ]
    )

    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for event in stream:
            assert not isinstance(event, events_.RunBlocked)
            if (
                isinstance(event, events_.HookEvent)
                and event.hook.status == "pending"
            ):
                assert event.hook.tool_call_id is None
                ai.resolve_hook("self_gate", {"approved": True})

    assert not stream.blocked


def _hook_event(hook: messages_.HookPart[Any]) -> events_.HookEvent:
    return events_.HookEvent(
        message=messages_.Message(role="internal", parts=[hook]), hook=hook
    )


def test_tracker_fold_sequence() -> None:
    """RunStateTracker is a pure fold usable over any event stream."""
    tracker = events_.RunStateTracker()
    two_calls = _multi_call_msg(("tc-1", "gated"), ("tc-2", "slow"))
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="h1",
        hook_type="ToolApproval",
        status="pending",
        tool_call_id="tc-1",
    )

    assert tracker.feed(events_.StreamStart()) is None
    # Parks mid-stream: suppressed while the model is producing.
    assert tracker.feed(_hook_event(hook)) is None
    # Stream over, but tc-2 is in flight and not blocked.
    assert tracker.feed(events_.StreamEnd(message=two_calls)) is None
    assert not tracker.blocked

    result = ai.tool_result(tool_call_id="tc-2", tool_name="slow", result="ok")
    transition = tracker.feed(result)
    assert isinstance(transition, events_.RunBlocked)
    assert [h.hook_id for h in transition.hooks] == ["h1"]
    assert tracker.blocked
    assert [h.hook_id for h in tracker.pending_hooks] == ["h1"]

    # The resolution flips the tracker back silently -- no mirror event.
    resolved = tracker.feed(
        _hook_event(hook.model_copy(update={"status": "resolved"}))
    )
    assert resolved is None
    assert not tracker.blocked


def test_message_aggregator_tolerates_block_events() -> None:
    agg = ai.agents.MessageAggregator()
    agg.feed(events_.RunBlocked())
    assert agg.snapshot().messages == ()
