"""Start a conversation FROM a history: the one way a conversation moves.

Same harness on another machine, or the other harness entirely — the call
is the same: read it as `history()`, start again with `session(history=)`.
Everything here runs on both harnesses at both locations; the cross-harness
tests run both directions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai.harnesses.experimental import Harness
from ai.types.events import ToolEnd
from ai.types.messages import Message, ReasoningPart, TextPart, ToolCallPart
from ai.workspaces.experimental import Local, Workspace
from tests.harnesses.experimental.conftest import SPECS, exact

pytestmark = pytest.mark.live

OTHER = {"claude": "codex", "codex": "claude"}


async def _source_history(
    kind: str, project: Path, prompt: str
) -> list[Message]:
    """A real conversation on THIS machine, read back as messages.

    Codex may write here: in its default read-only mode it declines a write
    without attempting one, and the history would carry no action at all.
    """
    options: dict[str, Any] = (
        {"sandbox": "workspace-write"} if kind == "codex" else {}
    )
    async with (
        Local(project) as ws,
        SPECS[kind](workspace=ws, **options) as harness,
    ):
        session = harness.session()
        await session.run(prompt)
        return await harness.history(session.session_id)


async def test_a_conversation_continues_on_any_machine(
    harness_kind: str, project: Path, any_harness: Harness
) -> None:
    """Start locally, continue wherever `any_harness` lives — same harness.

    The destination has never seen the conversation; it arrives as history.
    """
    history = await _source_history(
        harness_kind,
        project,
        "Remember this codeword: PLATYPUS. Reply with exactly: OK",
    )
    assert any("PLATYPUS" in _text(m) for m in history)

    session = any_harness.session(history=history)
    result = await session.run(
        "What codeword did I give you? Reply with the word only."
    )

    exact(result.text, "PLATYPUS")
    # The session's own past IS the history, from message one.
    assert [_text(m) for m in session.messages[: len(history)]] == [
        _text(m) for m in history
    ]


async def test_run_takes_a_history_too(
    harness_kind: str, project: Path, any_harness: Harness
) -> None:
    """Sugar: one prompt, on top of an earlier conversation."""
    history = await _source_history(
        harness_kind,
        project,
        "My favourite colour is VERMILION. Reply with exactly: NOTED",
    )
    result = await any_harness.run(
        "What is my favourite colour? Reply with the word only.",
        history=history,
    )
    exact(result.text, "VERMILION")


async def test_the_other_harness_picks_it_up(
    harness_kind: str, project: Path, any_workspace: Workspace
) -> None:
    """Claude -> Codex and Codex -> Claude, at both locations.

    The conversation is a list of messages; which harness wrote it does not
    matter.
    """
    history = await _source_history(
        harness_kind,
        project,
        "My cat is called MARMALADE. Reply with exactly: NOTED",
    )
    async with SPECS[OTHER[harness_kind]](workspace=any_workspace) as other:
        result = await other.run(
            "What is my cat called? Reply with the name only.", history=history
        )
    exact(result.text, "MARMALADE")


async def test_actions_travel_and_the_receiver_uses_its_own_tools(
    harness_kind: str, project: Path, any_workspace: Workspace
) -> None:
    """The history carries a tool call the receiver does not have.

    It must know what happened AND act with its own tools — never call the
    foreign one. Measured on both: Write became fileChange, fileChange became
    Edit.

    Two turns on purpose: codex labels mid-turn narration `commentary`, which
    the adapter keeps out of the settled message, so "recall then act" in one
    turn leaves only the act in `result.messages[-1]`. Recall is a turn of
    its own and read from `.text`.

    Paths in transferred actions are the SOURCE machine's — a `Write` to
    /Users/.../recipe.txt says nothing about where the file is now. Across
    machines the receiver is told, as any colleague would be.
    """
    history = await _source_history(
        harness_kind,
        project,
        "Create a file called recipe.txt containing exactly: two eggs. Then "
        "reply: OK",
    )
    foreign = {
        p.tool_name
        for m in history
        for p in m.parts
        if isinstance(p, ToolCallPart)
    }
    assert foreign, "the source must actually have used a tool"
    await any_workspace.write_text("recipe.txt", "two eggs\n")

    # A mode for codex only on this machine; in a sandbox the default writes.
    options: dict[str, Any] = (
        {"sandbox": "workspace-write"}
        if OTHER[harness_kind] == "codex" and any_workspace.owner != "provider"
        else {}
    )
    async with SPECS[OTHER[harness_kind]](
        workspace=any_workspace, **options
    ) as other:
        session = other.session(history=history)
        recalled = await session.run(
            "Earlier in this conversation a file was created. Which file, and "
            "what did " + "it contain? One line."
        )
        turn = session.stream(
            "That file now lives in the current working directory. Append the "
            "line " + "'second line' to it, then reply with exactly: DONE"
        )
        called = [
            e.tool_call.tool_name async for e in turn if isinstance(e, ToolEnd)
        ]

    assert (
        "recipe.txt" in recalled.text and "two eggs" in recalled.text
    ), recalled.text
    assert called, "the receiver must have acted"
    assert not (
        set(called) & foreign
    ), f"called a tool it does not have: {called}"
    assert "second line" in await any_workspace.read_text("recipe.txt")


async def test_reasoning_travels_where_the_destination_allows_it(
    any_harness: Harness,
) -> None:
    """The fact below exists NOWHERE but in a ReasoningPart.

    codex: arrives as a user-role handoff note and is used. Claude: cannot
    arrive at all — Anthropic's safeguards flag real reasoning replayed as
    user content ([reasoning_extraction]; measured with genuine Opus
    thinking) — so it is omitted, and the SDK says so with a warning rather
    than dropping context silently.
    """
    import warnings

    history = [
        Message(
            role="user",
            parts=[
                TextPart(
                    text="Help me plan the config layout for this service."
                )
            ],
        ),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(
                    text="The deploy region must stay eu-west-3 for GDPR; "
                    "I should keep that in mind."
                ),
                TextPart(
                    text="Sure — I'll use a single config.toml with "
                    "per-environment sections."
                ),
            ],
        ),
    ]
    ask = (
        "What deploy region did we settle on earlier, and why? "
        "If it was never said, reply UNKNOWN."
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await any_harness.run(ask, history=history)
    warned = [w for w in caught if "reasoning part" in str(w.message)]

    if any_harness.kind == "codex":
        assert "eu-west-3" in result.text.lower(), result.text
        assert not warned
    else:
        assert (
            "eu-west-3" not in result.text.lower()
        ), "claude must not have received the reasoning"
        assert warned, "omitting context silently is not allowed"
        assert "1 reasoning part" in str(warned[0].message)


def _text(m: Message) -> str:
    return "".join(p.text for p in m.parts if isinstance(p, TextPart))
