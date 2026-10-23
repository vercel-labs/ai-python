"""Every harness, driven inside a microVM instead of on this machine.

The contract is that nothing about using the SDK changes: the same
factories, the same turns, the same results. What changes is where the
process lives — and these tests exist to prove that difference is
invisible.
"""

from __future__ import annotations

import pytest

from ai.harnesses.experimental import Allow, ApprovalContext, Decision
from ai.types.events import TextDelta
from ai.types.messages import Message
from ai.workspaces.experimental import Workspace
from tests.harnesses.experimental.conftest import SPECS, exact

# Remote tests pay for a sandbox boot and a harness install before the
# first token, so the local 300s budget is not the right yardstick.
pytestmark = [
    pytest.mark.live,
    pytest.mark.sandbox,
    pytest.mark.slow,
    pytest.mark.timeout(900),
]


async def test_a_turn_runs_in_the_microvm(
    harness_kind: str, remote_workspace: Workspace
) -> None:
    await remote_workspace.write_text(
        "util.py", "def add(a, b):\n    return a + b\n"
    )

    async with SPECS[harness_kind](workspace=remote_workspace) as harness:
        assert harness.version, "the CLI was found or installed in the VM"
        result = await harness.run("Reply with exactly: READY")

    exact(result.text, "READY")
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens > 0


async def test_the_agent_reads_the_remote_workspace(
    harness_kind: str, remote_workspace: Workspace
) -> None:
    """The files it sees are the workspace's, not this machine's."""
    await remote_workspace.write_text(
        "util.py", "def multiply(a, b):\n    return a * b\n"
    )

    async with SPECS[harness_kind](workspace=remote_workspace) as harness:
        result = await harness.run(
            "Read util.py in the current directory and reply with the "
            + "function name only."
        )

    exact(result.text, "multiply")


async def test_the_agent_writes_to_the_remote_workspace(
    harness_kind: str, remote_workspace: Workspace
) -> None:
    async def approve(ctx: ApprovalContext) -> Decision:
        return Allow()

    async with SPECS[harness_kind](
        workspace=remote_workspace, approve=approve
    ) as harness:
        await harness.run(
            "Write the word HELLO into greeting.txt, then reply DONE."
        )

    assert await remote_workspace.exists("greeting.txt")
    assert "HELLO" in await remote_workspace.read_text("greeting.txt")


async def test_streaming_works_over_the_wire(
    harness_kind: str, remote_workspace: Workspace
) -> None:
    async with SPECS[harness_kind](workspace=remote_workspace) as harness:
        session = harness.session()
        turn = session.stream("Reply with exactly: STREAMED")
        streamed = "".join(
            [e.chunk async for e in turn if isinstance(e, TextDelta)]
        )

    assert "STREAMED" in streamed
    assert streamed.strip() == turn.result.text.strip()


async def test_a_conversation_remembers_across_turns(
    harness_kind: str, remote_workspace: Workspace
) -> None:
    async with SPECS[harness_kind](workspace=remote_workspace) as harness:
        session = harness.session()
        await session.run("Remember this codeword: PLATYPUS. Reply OK.")
        answer = await session.run(
            "What codeword did I give you? Reply with the word only."
        )

    assert "PLATYPUS" in answer.text.upper()


async def test_sessions_are_the_harness_own_wherever_it_runs(
    harness_kind: str, remote_workspace: Workspace
) -> None:
    """The conversation store lives WHERE THE HARNESS RUNS, and listing,
    reading and forking must read it there.

    Measured before the fix: claude's `sessions()` returned `[]` for a
    conversation that had just happened in the VM, because the adapter read
    this machine's `~/.claude/projects` for a `/vercel/sandbox` project.
    Codex was unaffected — its listing is an RPC into the VM.
    """
    async with SPECS[harness_kind](workspace=remote_workspace) as harness:
        session = harness.session()
        await session.run(
            "Remember this codeword: PLATYPUS. Reply with exactly: OK"
        )
        sid = session.session_id

        listed = await harness.sessions()
        assert sid in {s.session_id for s in listed}, [
            s.session_id for s in listed
        ]
        # Each entry says whose it is, so a list gathered across harnesses
        # is self-describing.
        assert all(s.kind == harness.kind for s in listed)

        history = await harness.history(sid)
        assert any("PLATYPUS" in _text(m) for m in history), history

        branch = await harness.fork(sid)
        assert branch.session_id != sid
        result = await branch.run(
            "What codeword did I give you? Reply with the word only."
        )

    assert "PLATYPUS" in result.text.upper()


def _text(message: Message) -> str:
    from ai.types.messages import TextPart

    return "".join(p.text for p in message.parts if isinstance(p, TextPart))
