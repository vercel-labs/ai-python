"""A program on a pseudo-terminal, identically at both locations.

The shell checks use `sh -i` because a shell answers `stty size` with what the
program actually sees. The TUI checks use the real harness TUIs: the contract
is that they render — a terminal escape sequence appears — not what they say.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from ai.harnesses.experimental import Harness, SessionInfo
from ai.workspaces.experimental import Workspace
from ai.workspaces.experimental._pty import Pty
from ai.workspaces.experimental.errors import WorkspaceError
from tests.harnesses.experimental.conftest import SPECS, requires_claude

READY = {"claude-code": "\u276f".encode(), "codex": b"Ask Codex"}
"""What each TUI's composer shows once it accepts input: claude's input
caret (a footer string it used to print changed between two releases in
one day — markers must be structure, not wording), codex's composer hint.
Quiet-period detection is not enough: codex's splash animates without
pause, and claude pauses mid-startup — keystrokes typed before these
markers are lost."""


async def _ready(pty: Pty, kind: str, timeout: float = 150) -> bytes:
    """Read until the TUI shows its ready marker, then give it a beat."""
    seen = await _read_until(pty, READY[kind], timeout)
    assert (
        READY[kind] in seen
    ), f"the {kind} TUI never reached its prompt; saw {seen[-200:]!r}"
    await asyncio.sleep(2)
    return seen


async def _settle(pty: Pty, quiet: float = 2.0, timeout: float = 60) -> bytes:
    """Read until the program has been silent for `quiet` seconds — a TUI
    is ready for input when it has stopped drawing, whatever it draws."""
    seen = b""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            async with asyncio.timeout(quiet):
                chunk = await pty.receive()
        except TimeoutError:
            if seen:
                return seen
            continue
        if not chunk:
            break
        seen += chunk
    return seen


async def _read_until(pty: Pty, needle: bytes, timeout: float = 10) -> bytes:
    seen = b""
    async with asyncio.timeout(timeout):
        while needle not in seen:
            chunk = await pty.receive()
            if not chunk:
                break
            seen += chunk
    return seen


async def test_a_pty_is_a_terminal_to_the_program(
    any_workspace: Workspace,
) -> None:
    pty = await any_workspace.pty(["sh", "-i"], size=(80, 24))
    try:
        await pty.send(b"echo marker-$((6*7))\n")
        out = await _read_until(pty, b"marker-42")
        assert b"marker-42" in out
        await pty.resize(120, 50)
        await asyncio.sleep(0.3)
        await pty.send(b"stty size\n")
        out = await _read_until(pty, b"50 120")
        assert b"50 120" in out, "the program must see the new size"
    finally:
        await pty.close()
    await pty.wait()  # settles; must not hang


async def test_an_unnamed_pty_ends_with_the_connection(
    any_workspace: Workspace,
) -> None:
    pty = await any_workspace.pty(["sh", "-i"])
    await pty.send(b"echo up\n")
    await _read_until(pty, b"up")
    await pty.detach()
    # Nothing to attach to: the program went with the connection.
    assert pty.name is None


async def test_a_named_pty_survives_detach_and_replays_on_attach(
    any_workspace: Workspace,
) -> None:
    name = f"t-{uuid.uuid4().hex[:8]}"
    pty = await any_workspace.pty(["sh", "-i"], name=name)
    await pty.send(b"echo before-detach\n")
    await _read_until(pty, b"before-detach")
    await pty.detach()
    assert name in [
        p.name for p in await any_workspace.ptys()
    ], "a named pty is listed while it runs"

    again = await any_workspace.attach(name)
    try:
        replay = await _read_until(again, b"before-detach", timeout=15)
        assert (
            b"before-detach" in replay
        ), "attach must replay what happened while detached"
        await again.send(b"echo after-attach\n")
        assert b"after-attach" in await _read_until(again, b"after-attach")
    finally:
        await again.close()
    await asyncio.sleep(0.5)
    assert name not in [
        p.name for p in await any_workspace.ptys()
    ], "close() ends it, and the listing says so"
    with pytest.raises(WorkspaceError):
        await any_workspace.attach(name)


async def test_listing_ptys_never_takes_one_from_its_client(
    any_workspace: Workspace,
) -> None:
    """A pty serves one client and a new connection displaces it, so the
    listing must learn liveness and attachment without connecting. It says
    whether a client is on each pty, and asking leaves that client attached."""
    name = f"listing-{uuid.uuid4().hex[:8]}"
    p = await any_workspace.pty(["sh", "-i"], name=name)
    try:
        listed = {i.name: i for i in await any_workspace.ptys()}
        assert (
            name in listed and listed[name].attached
        ), "a pty with a client on it is listed as attached"
        assert listed[name].pid == p.pid
        for _ in range(3):
            await any_workspace.ptys()
        await p.send(b"echo after-listing-$((6*7))\n")
        seen = b""
        async with asyncio.timeout(20):
            while b"after-listing-42" not in seen:
                chunk = await p.receive()
                assert chunk, "listing displaced the attached client"
                seen += chunk
        await p.detach()
        async with asyncio.timeout(10):
            while True:
                listed = {i.name: i for i in await any_workspace.ptys()}
                if name in listed and not listed[name].attached:
                    break
                await asyncio.sleep(0.2)
    finally:
        again = await any_workspace.attach(name)
        await again.close()


async def test_a_program_that_ends_before_anyone_attaches_still_reports(
    any_workspace: Workspace,
) -> None:
    """The holder opens its socket before the program starts and, if the
    program ends before a client has connected, waits briefly for one: a
    short program's output and status reach whoever started it."""
    p = await any_workspace.pty(["sh", "-c", "echo over-before-attach; exit 7"])
    seen = b""
    while True:
        chunk = await p.receive()
        if not chunk:
            break
        seen += chunk
    status = await p.wait()
    assert (
        b"over-before-attach" in seen
    ), f"the output was lost; the terminal saw {seen!r}, status {status!r}"
    assert (
        status == 7
    ), f"the exit status was lost: {status!r}; the terminal saw {seen!r}"


@pytest.mark.live
async def test_the_harness_tui_renders(any_harness: Harness) -> None:
    tui = await any_harness.tui(size=(120, 40))
    pty = tui.pty
    try:
        seen = b""
        async with asyncio.timeout(60):
            while b"\x1b[" not in seen:
                chunk = await pty.receive()
                if not chunk:
                    break
                seen += chunk
        assert (
            b"\x1b[" in seen
        ), f"the TUI never drew a screen; got {seen[:200]!r}"
    finally:
        await tui.close()


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.sandbox
@requires_claude
async def test_a_claude_tui_session_is_discoverable_and_locked(
    remote_workspace: Workspace,
) -> None:
    """A TUI conversation is a conversation.

    Claude runs under the id we minted, so from launch it holds the session
    lock; after its first turn it appears in sessions() marked running, and a
    second client is refused — the same single-writer rule as any other way of
    driving it. (Codex mints its own id, so a new codex TUI is discoverable but
    not pre-locked; that is a separate follow-up.) Sandbox only: completing
    first-run setup is the SDK's job on a machine it owns, never on a user's
    own.
    """
    any_workspace = remote_workspace
    from ai.harnesses.experimental import claude_code
    from ai.harnesses.experimental.errors import SessionBusyError

    agent = claude_code(workspace=any_workspace)
    await agent.open()
    tui = await agent.tui(size=(120, 40))
    pty = tui.pty
    assert (
        tui.session_id is not None and tui.handle is not None
    ), "claude chose the id: known at launch"
    try:
        # Type only once the TUI shows its prompt: keystrokes sent while it is
        # still starting up are lost (measured — and the reason an eager
        # version of this test timed out).
        await _ready(pty, "claude-code")
        await pty.send(b"Reply with exactly OK\r")

        running: list[SessionInfo] = []
        async with asyncio.timeout(150):
            while not running:
                await asyncio.sleep(3)
                running = [i for i in await agent.sessions() if i.running]
        other = claude_code(workspace=any_workspace)
        await other.open()
        try:
            with pytest.raises(SessionBusyError):
                await other.resume(running[0].session_id)
        finally:
            await other.close()
    finally:
        await tui.close()
        await agent.close()


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.sandbox
async def test_codex_resume_in_the_tui_shows_the_conversation_and_is_handed_over(  # noqa: E501
    remote_workspace: Workspace,
) -> None:
    """`codex resume <id>` in codex's own TUI, measured end to end at the
    level that is deterministic: the resumed TUI renders the conversation's
    history, and because this client held the thread headless, `tui()` hands
    it over first — codex enforces one writer across processes and would
    otherwise block the TUI with "this conversation is open in another app".

    Typing into codex's TUI is deliberately NOT automated here: in a pty
    without a real terminal negotiating with it, codex echoes neither plain
    nor kitty-encoded keys (measured), and only a human at `tty.bridge`
    settles that. A human check is one command:
    `python examples/harnesses/tui.py codex`.
    """
    import re

    from ai.harnesses.experimental import codex

    ansi = re.compile(
        rb"\x1b\[[0-9;?<>=]*[ "
        rb"-/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\\\)|\x1b[=>78]|\x1b\([B0]"
    )
    agent = codex(workspace=remote_workspace)
    await agent.open()
    session = agent.session()
    await session.run("Remember the codeword KUMQUAT. Reply with exactly: OK")
    sid = session.session_id

    tui = await agent.tui(sid, size=(120, 40))
    assert tui.session_id == sid and tui.handle is not None
    try:
        screen = await _read_until(tui.pty, b"Ask Codex", timeout=150)
        shown = ansi.sub(b"", screen)
        assert b"Ask Codex" in shown, "the resumed TUI never reached its prompt"
        assert (
            b"KUMQUAT" in shown
        ), "the resumed TUI must render the conversation it resumed"
        assert (
            b"open in another app" not in shown
        ), "hand-over must release codex's cross-process hold"
        assert (
            getattr(agent.adapter, "_peer", None) is None
        ), "hand-over stops this client's app-server"
    finally:
        await tui.close()
    # The headless side relaunches lazily and still lists the conversation.
    assert sid in {i.session_id for i in await agent.sessions()}
    await agent.close()


@pytest.mark.live
async def test_tui_refuses_both_a_session_and_a_history(
    any_harness: Harness,
) -> None:
    with pytest.raises(ValueError, match="not both"):
        await any_harness.tui("some-id", history=[])


@pytest.mark.live
@pytest.mark.slow
async def test_the_tui_starts_from_a_history(
    harness_kind: str, any_workspace: Workspace
) -> None:
    """`tui(history=...)` means what `session(history=...)` means: a new
    conversation that begins with this past — the way a conversation is
    pushed into a sandbox TUI, or brought home into a local one. Nothing
    headless is started to stage it, and the TUI is its only writer.

    Observation only, no typing. At BOTH locations: the TUI is up on the
    staged conversation, which claude lists and holds (codex does not list
    a thread until its first turn — measured), and a second client is
    refused. Reaching the prompt and redrawing the past are asserted only
    where the SDK owns first-run setup (a provider-owned workspace): on a
    user's own machine a fresh directory opens on claude's trust dialog,
    which is the human's to answer, never the SDK's.
    """
    from ai.harnesses.experimental.errors import (
        ResumeFailedError,
        SessionBusyError,
    )

    agent = SPECS[harness_kind](workspace=any_workspace)
    await agent.open()
    source = agent.session()
    await source.run("Remember the codeword PAPAYA. Reply with exactly: OK")
    history = await agent.history(source.session_id)
    await source.close()

    tui = await agent.tui(history=history, size=(120, 40))
    assert (
        tui.session_id is not None
    ), "a conversation with a past has an id at launch, on both harnesses"
    assert tui.handle is not None and tui.handle.session_id == tui.session_id
    assert (
        tui.session_id != source.session_id
    ), "a NEW conversation, not the source"
    try:
        if any_workspace.owner == "provider":
            screen = await _read_until(tui.pty, READY[agent.kind], timeout=150)
            assert (
                READY[agent.kind] in screen
            ), "the TUI must reach its prompt on the staged conversation"
            if agent.kind == "claude-code":
                assert (
                    b"PAPAYA" in screen
                ), "claude redraws the resumed transcript"
        else:
            drawn = await _read_until(tui.pty, b"\x1b[", timeout=60)
            assert (
                b"\x1b[" in drawn
            ), "the TUI must come up on the staged conversation"
        if agent.kind == "claude-code":
            running = {
                i.session_id for i in await agent.sessions() if i.running
            }
            assert (
                tui.session_id in running
            ), "the staged conversation is listed and held by the TUI"
        # One writer, on both harnesses: a second client cannot take it over.
        other = SPECS[harness_kind](workspace=any_workspace)
        await other.open()
        try:
            with pytest.raises((SessionBusyError, ResumeFailedError)):
                await other.resume(tui.session_id)
        finally:
            await other.close()
    finally:
        await tui.close()
        await agent.close()
