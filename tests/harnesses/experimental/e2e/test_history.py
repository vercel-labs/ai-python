"""Reading and taking over conversations — including ones we never created.

The distinguishing capability: a harness's sessions are its own, and they
outlive us. A conversation someone started in their terminal is readable,
adoptable, and forkable from here, with the SAME message type as a session
we drove ourselves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ai.harnesses.experimental import Allow, Harness
from ai.types.messages import Message, TextPart, ToolCallPart
from tests.harnesses.experimental.conftest import requires

pytestmark = pytest.mark.live


async def test_sessions_are_listed_for_the_workspace(harness: Harness) -> None:
    session = harness.session()
    await session.run("Reply with exactly: LISTED")

    infos = await harness.sessions()

    assert any(i.session_id == session.session_id for i in infos)
    found = next(i for i in infos if i.session_id == session.session_id)
    assert found.updated_at is not None


async def test_history_returns_ai_sdk_messages(harness: Harness) -> None:
    """One message type across the whole SDK.

    A replayed transcript and a live one are the same shape — v1's two read
    paths drifted apart and this is the assertion that stops it happening again.
    """
    session = harness.session()
    await session.run("Remember this codeword: PLATYPUS. Reply OK.")

    history = await harness.history(session.session_id)

    assert history and all(isinstance(m, Message) for m in history)
    assert any(m.role == "user" for m in history)
    assert any(m.role == "assistant" for m in history)
    assert "PLATYPUS" in " ".join(
        p.text for m in history for p in m.parts if isinstance(p, TextPart)
    )


async def test_history_is_full_fidelity_not_a_summary(
    make_harness: Callable[..., Any],
) -> None:
    """Tool calls survive the round trip.

    Both harnesses can serve complete records — claude's stored messages keep
    every content block, codex's itemsView='full' keeps every ThreadItem — so
    nothing here settles for a display summary.

    The prompt names a shell command on purpose: asking a model to "read a
    file" does not guarantee a tool call, and codex answered one such
    question with no recorded tool item at all.
    """

    async def approve(ctx: Any) -> Allow:
        return Allow()

    async with make_harness(approve=approve, writable=True) as harness:
        session = harness.session()
        # A write, not a read: codex answered "read README.md" with no
        # recorded tool item at all, so a read proves nothing about replay.
        await session.run(
            "Write the word HELLO into greeting.txt, then reply DONE."
        )
        history = await harness.history(session.session_id)

    calls = [p for m in history for p in m.parts if isinstance(p, ToolCallPart)]
    assert (
        calls
    ), "a replayed transcript must keep the tool calls, not just the prose"
    assert calls[0].tool_name
    assert any(m.role == "tool" for m in history), "tool results survive too"


async def test_history_paginates(harness: Harness) -> None:
    session = harness.session()
    await session.run("Reply with exactly: ONE")
    await session.run("Reply with exactly: TWO")

    first = await harness.history(session.session_id, limit=1)
    everything = await harness.history(session.session_id)

    assert len(first) == 1
    assert len(everything) > len(first)


async def test_adopt_takes_over_a_session_we_did_not_create(
    harness: Harness,
) -> None:
    """The headline: continue a conversation this process never started."""
    requires(harness, "resume")
    stranger = harness.session()
    await stranger.run("Remember this codeword: PLATYPUS. Reply OK.")
    session_id = stranger.session_id

    resumed = await harness.resume(session_id)

    assert resumed.session_id == session_id
    assert (
        resumed.messages
    ), "a resumed session knows its past before you prompt it"
    answer = await resumed.run(
        "What codeword did I give you? Reply with the word only."
    )
    assert "PLATYPUS" in answer.text.upper()


async def test_fork_branches_instead_of_colliding(harness: Harness) -> None:
    """The safe way to pick up someone's live conversation: branch it, so two
    writers never share one transcript."""
    requires(harness, "fork")
    original = harness.session()
    await original.run("Remember this codeword: PLATYPUS. Reply OK.")

    branch = await harness.fork(original.session_id)

    assert (
        branch.session_id != original.session_id
    ), "a fork is a new conversation"
    answer = await branch.run(
        "What codeword did I give you? Reply with the word only."
    )
    assert "PLATYPUS" in answer.text.upper(), "the fork inherits the past"

    # The original is untouched by anything the branch does.
    still = await original.run("Reply with exactly: ORIGINAL")
    assert "ORIGINAL" in still.text


async def test_history_of_an_unknown_session_raises(harness: Harness) -> None:
    with pytest.raises(Exception, match=r"(?i)session"):
        await harness.history("00000000-0000-0000-0000-000000000000")


async def test_history_pages_with_limit_and_offset(
    any_harness: Harness,
) -> None:
    """`offset=` had no test at all; `limit=` one.

    A page is a window onto the same ordered transcript, wherever the harness
    keeps it.
    """
    session = any_harness.session()
    await session.run("Reply with exactly: ONE")
    await session.run("Reply with exactly: TWO")
    await session.run("Reply with exactly: THREE")

    everything = await any_harness.history(session.session_id)
    assert (
        len(everything) >= 6
    ), "three user + three assistant messages at least"
    first_two = await any_harness.history(session.session_id, limit=2)
    next_two = await any_harness.history(session.session_id, limit=2, offset=2)

    # Compare by content, not id: `history()` reconstructs messages on every
    # call and mints fresh ids each time (neither adapter derives them from
    # the transcript — a real gap, tracked separately). The three prompts
    # are distinct, so (role, text) identifies each message unambiguously.
    def shape(ms: list[Message]) -> list[tuple[str, str]]:
        return [
            (
                m.role,
                "".join(p.text for p in m.parts if isinstance(p, TextPart)),
            )
            for m in ms
        ]

    assert shape(first_two) == shape(everything[:2])
    assert shape(next_two) == shape(everything[2:4])
    assert shape(first_two) != shape(next_two)
