"""Steering and stopping a running agent — the reason this SDK exists.

`agent()`-style call-and-wait APIs cannot express any of this: once the
prompt is sent you wait for whatever comes back. Here the turn is open for
writing while it runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from ai.harnesses.experimental import Allow, Harness
from ai.types.events import TextDelta, ToolStart
from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental.conftest import requires


async def allow_everything(ctx: Any) -> Allow:
    """Steering is only observable if the agent can actually act: without a
    hook the harness's own policy governs, and a denied write means there is
    no work in flight to redirect."""
    return Allow()


pytestmark = [pytest.mark.live, pytest.mark.slow]


async def test_steer_changes_course_mid_turn(
    any_make_harness: Callable[..., Any], any_workspace: Workspace
) -> None:
    async with any_make_harness(
        approve=allow_everything, writable=True
    ) as any_harness:
        requires(any_harness, "steer")
        await _steer_mid_turn(any_harness, any_workspace)


async def _steer_mid_turn(
    any_harness: Harness, any_workspace: Workspace
) -> None:
    session = any_harness.session()

    turn = session.stream(
        "Create files step1.txt, step2.txt ... step40.txt, one at a time, "
        + "each containing its own number. Work slowly and do not stop early."
    )

    steered = False
    async for event in turn:
        if isinstance(event, ToolStart) and not steered:
            # The agent is demonstrably working; redirect it now.
            await session.steer(
                "STOP creating numbered files. Instead write the single word "
                + "STEERED into steered.txt, then finish."
            )
            steered = True

    assert steered, "the turn never started work, so nothing was steered"
    # What is guaranteed is that the agent ACTS on the new instruction
    # within the same turn. Whether it abandons the work already in flight
    # is not: measured over three runs each, Claude stopped early 3/3 but
    # Codex only 1/3 — twice it finished all forty files first and obeyed
    # after. Asserting preemption here passes on luck. `stop()` is the verb
    # that halts immediately; `steer()` redirects.
    assert await any_workspace.exists("steered.txt")
    assert "STEERED" in await any_workspace.read_text("steered.txt")


async def test_steer_is_recorded_in_the_conversation(
    any_harness: Harness,
) -> None:
    """A steering message is a real user message, not an invisible side
    channel."""
    requires(any_harness, "steer")
    session = any_harness.session()

    turn = session.stream(
        "Write the numbers 1 through 2000 separated by newlines, typing them "
        + "out yourself. Output nothing else."
    )
    async for _ in turn:
        await session.steer(
            "Actually, stop counting and reply with exactly: HALTED"
        )
        break
    async for _ in turn:
        pass

    user_text = " ".join(m.text for m in session.messages if m.role == "user")
    assert "HALTED" in user_text, "steering must be visible in the transcript"


async def test_steer_without_an_active_turn_raises(
    any_harness: Harness,
) -> None:
    requires(any_harness, "steer")
    session = any_harness.session()

    with pytest.raises(RuntimeError, match="no active turn"):
        await session.steer("nothing is running")


async def test_stop_settles_the_turn_as_cancelled(any_harness: Harness) -> None:
    requires(any_harness, "stop")
    session = any_harness.session()

    turn = session.stream(
        "Write the numbers 1 through 2000 separated by newlines, typing them "
        + "out yourself. Output nothing else."
    )
    async for _ in turn:
        await session.stop()
        break

    # stop() settles the turn: no second iteration needed, and abandoning
    # the stream must not wedge the session.
    assert turn.result.finish_reason == "cancelled"


async def test_stop_leaves_the_conversation_resumable(
    any_harness: Harness,
) -> None:
    """Stopping is graceful: the any_harness keeps its transcript and we
    continue."""
    requires(any_harness, "stop")
    session = any_harness.session()

    turn = session.stream(
        "Write the numbers 1 through 2000 separated by newlines, typing them "
        + "out yourself. Output nothing else."
    )
    # Stop only once the agent has actually SAID something. Interrupting on
    # the very first event lands before the turn is committed, and there is
    # then genuinely nothing for the any_harness to remember.
    async for event in turn:
        if isinstance(event, TextDelta):
            await session.stop()
            break

    resumed = await session.run(
        "What did I just ask you to write? Reply in one short sentence."
    )
    assert "2000" in resumed.text or "number" in resumed.text.lower()


async def test_external_cancellation_still_propagates(
    any_harness: Harness,
) -> None:
    """`stop()` settles; asyncio.timeout raises.

    Both, unchanged — the stdlib contract is not quietly swallowed by ours.

    The prompt must genuinely outlive the deadline; "count slowly" does not,
    because the model shells out and answers in seconds.
    """
    session = any_harness.session()

    # Five seconds: long enough for the turn to be genuinely under way,
    # short enough that no model finishes 2000 numbers inside it. A
    # borderline deadline makes this test flaky, not rigorous.
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(5):
            await session.run(
                "Write the numbers 1 through 2000 separated by newlines, "
                + "typing them out yourself. Output nothing else."
            )
