"""`tty.bridge` needs a real terminal, so give it one: a pseudo-terminal of
its own, with the bridge running inside it and this test playing the human."""

from __future__ import annotations

import contextlib
import os
import pty
import select
import sys
import time
import uuid
from pathlib import Path

import pytest

DRIVER = """
import asyncio, sys
from ai.workspaces.experimental import tty
from ai.workspaces.experimental import Local
async def main():
    async with Local(sys.argv[1]) as ws:
        p = await ws.pty(["sh", "-i"], name=sys.argv[2])
        status = await tty.bridge(p)
        print(f"BRIDGE-RETURNED:{status}", flush=True)
asyncio.run(main())
"""


def _read_until(fd: int, needle: bytes, timeout: float = 15) -> bytes:
    seen, end = b"", time.monotonic() + timeout
    while needle not in seen and time.monotonic() < end:
        ready, _, _ = select.select([fd], [], [], 0.5)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            seen += chunk
    return seen


@pytest.mark.timeout(60)
def test_bridge_types_through_and_detaches_on_the_key(tmp_path: Path) -> None:
    name = f"bridge-{uuid.uuid4().hex[:8]}"
    pid, master = pty.fork()
    if pid == 0:  # the "human's terminal": run the driver inside it
        os.chdir(tmp_path)
        os.execv(
            sys.executable, [sys.executable, "-c", DRIVER, str(tmp_path), name]
        )
    try:
        # Wait for the inner shell's prompt to come through the bridge, as a
        # human would, before typing at it.
        assert b"$ " in _read_until(
            master, b"$ "
        ), "the bridged shell never showed a prompt"
        os.write(master, b"echo typed-$((6*7))\n")
        out = _read_until(master, b"typed-42")
        assert (
            b"typed-42" in out
        ), "keystrokes must reach the program and its output must come back"
        os.write(master, b"\x1d")  # Ctrl-]: detach
        out = _read_until(master, b"BRIDGE-RETURNED:")
        assert (
            b"BRIDGE-RETURNED:None" in out
        ), "the detach key returns None, not an exit status"
        _, raw = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(raw) == 0
    finally:
        os.close(master)

    # The named program outlived the bridge; attach finds it and close() ends
    # it.
    import asyncio

    from ai.workspaces.experimental import Local

    async def check() -> None:
        async with Local(tmp_path) as ws:
            assert name in [p.name for p in await ws.ptys()]
            again = await ws.attach(name)
            await again.close()
            await asyncio.sleep(0.5)
            assert name not in [p.name for p in await ws.ptys()]

    asyncio.run(check())


def _wait_exit(pid: int, timeout: float = 15) -> int | None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        done, raw = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(raw)
        time.sleep(0.1)
    return None


@pytest.mark.timeout(60)
def test_bridge_ends_when_the_terminal_goes_away(tmp_path: Path) -> None:
    """The failure that once left a real terminal stuck: the bridge's reader
    stops working while the program keeps drawing. Closing the terminal end
    must END the bridge (tty restored) rather than hang in raw mode."""
    name = f"bridge-{uuid.uuid4().hex[:8]}"
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execv(
            sys.executable, [sys.executable, "-c", DRIVER, str(tmp_path), name]
        )
    assert b"$ " in _read_until(
        master, b"$ "
    ), "the bridged shell never showed a prompt"
    os.close(master)  # the human's terminal vanishes
    status = _wait_exit(pid)
    assert (
        status is not None
    ), "the bridge must end when its terminal is gone, not hang"
    import asyncio

    from ai.workspaces.experimental import Local

    async def cleanup() -> None:
        async with Local(tmp_path) as ws:
            if name in [p.name for p in await ws.ptys()]:
                await (await ws.attach(name)).close()

    asyncio.run(cleanup())


@pytest.mark.timeout(60)
def test_bridge_ends_when_another_client_attaches(tmp_path: Path) -> None:
    """A named pty serves one client; a second attach displaces the first.

    The displaced bridge must return (with a reason), leaving the program
    running for the newcomer.
    """
    import asyncio

    from ai.workspaces.experimental import Local

    name = f"bridge-{uuid.uuid4().hex[:8]}"
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execv(
            sys.executable, [sys.executable, "-c", DRIVER, str(tmp_path), name]
        )
    try:
        assert b"$ " in _read_until(master, b"$ ")

        async def displace() -> None:
            async with Local(tmp_path) as ws:
                other = await ws.attach(name)  # displaces the bridge's client
                await other.send(b"echo still-alive\n")
                seen = b""
                async with asyncio.timeout(10):
                    while b"still-alive" not in seen:
                        chunk = await other.receive()
                        if not chunk:
                            break
                        seen += chunk
                assert (
                    b"still-alive" in seen
                ), "the program must keep running for the new client"
                await other.close()

        asyncio.run(displace())
        out = _read_until(master, b"BRIDGE-RETURNED:")
        assert b"BRIDGE-RETURNED:None" in out
        assert b"another client attached" in out
        assert _wait_exit(pid) == 0
    finally:
        with contextlib.suppress(OSError):
            os.close(master)


BURST_DRIVER = """
import asyncio, sys
from ai.workspaces.experimental import tty
from ai.workspaces.experimental import Local
async def main():
    async with Local(sys.argv[1]) as ws:
        # A program that, once the bridge is up, redraws a long transcript at
        # once: 4096 numbered lines, then exits.
        p = await ws.pty([
            "sh",
            "-c",
            "sleep 1; i=0; while [ $i -lt 4096 ]; do echo "
            "\\"line-$i-" + "x" * 60 + "\\"; i=$((i+1)); done",
        ])
        status = await tty.bridge(p)
        print(f"BRIDGE-RETURNED:{status}", flush=True)
asyncio.run(main())
"""


@pytest.mark.timeout(90)
def test_bridge_delivers_a_burst_to_a_slow_terminal(tmp_path: Path) -> None:
    """What stuck a real terminal: a long transcript redraw on a terminal that
    drains slower than the program writes. Every byte must arrive, in order,
    and the bridge must return — nothing truncated, nothing failing."""
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execv(
            sys.executable, [sys.executable, "-c", BURST_DRIVER, str(tmp_path)]
        )
    try:
        time.sleep(
            2
        )  # the human's terminal is busy: nothing drained while the burst lands
        seen = b""
        end = time.monotonic() + 60
        while b"BRIDGE-RETURNED:" not in seen and time.monotonic() < end:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                try:
                    chunk = os.read(master, 512)  # and then drains slowly
                except OSError:
                    break
                if not chunk:
                    break
                seen += chunk
                time.sleep(0.002)
        text = seen.replace(b"\r\n", b"\n").decode(errors="replace")
        lines = [ln for ln in text.split("\n") if ln.startswith("line-")]
        assert (
            len(lines) == 4096
        ), f"{len(lines)} of 4096 lines reached the terminal"
        expected = [f"line-{i}-" + "x" * 60 for i in range(4096)]
        assert lines == expected, "every line must arrive whole and in order"
        assert "BRIDGE-RETURNED:0" in text
        assert _wait_exit(pid) == 0
    finally:
        with contextlib.suppress(OSError):
            os.close(master)


QUICK_DRIVER = """
import asyncio, sys
from ai.workspaces.experimental import tty
from ai.workspaces.experimental import Local
async def main():
    async with Local(sys.argv[1]) as ws:
        p = await ws.pty(["sh", "-c", "echo quick-output; exit 3"])
        status = await tty.bridge(p)
        print(f"BRIDGE-RETURNED:{status}", flush=True)
asyncio.run(main())
"""


@pytest.mark.timeout(60)
def test_bridge_shows_a_program_that_ended_before_it_was_ready(
    tmp_path: Path,
) -> None:
    """A program can finish before the bridge has even sized its terminal.

    Its output and exit status must still come through, not a traceback from a
    resize sent to a program that is gone.
    """
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execv(
            sys.executable, [sys.executable, "-c", QUICK_DRIVER, str(tmp_path)]
        )
    try:
        out = _read_until(master, b"BRIDGE-RETURNED:")
        assert (
            b"quick-output" in out
        ), "the program's output must reach the terminal"
        assert (
            b"BRIDGE-RETURNED:3" in out
        ), "the program's exit status is the bridge's return"
        assert b"Traceback" not in out
        assert (
            b"[tty.bridge]" not in out
        ), "a program that simply ended is not a failure to report"
        assert _wait_exit(pid) == 0
    finally:
        with contextlib.suppress(OSError):
            os.close(master)


def test_the_modes_a_program_set_are_undone_and_nothing_else() -> None:
    """What claude leaves on when you detach from it, measured, plus the
    enhanced keyboard protocol it asks for: each is switched off, and only
    what the program itself set."""
    from ai.workspaces.experimental.tty import _TerminalModes

    modes = _TerminalModes()
    modes.feed(b"\x1b[?25l\x1b[?2004h\x1b[?1004h\x1b[?2031h")
    modes.feed(
        b"hello \x1b[>1u world \x1b[>5u"
    )  # two keyboard-protocol levels pushed
    modes.feed(b"\x1b[?1000h\x1b[?1000l")  # set and cleared: nothing to undo
    restore = modes.restore()
    for undo in (
        b"\x1b[?25h",
        b"\x1b[?2004l",
        b"\x1b[?1004l",
        b"\x1b[?2031l",
        b"\x1b[<2u",
    ):
        assert undo in restore, undo
    assert (
        b"1000" not in restore
    ), "a mode the program already cleared is not touched"
    assert (
        _TerminalModes().restore() == b""
    ), "a program that set nothing leaves nothing to undo"


def test_a_sequence_split_across_reads_is_still_seen() -> None:
    from ai.workspaces.experimental.tty import _TerminalModes

    modes = _TerminalModes()
    modes.feed(b"text \x1b[?20")
    modes.feed(b"04h more \x1b")
    modes.feed(b"[>1u")
    assert modes.restore() == b"\x1b[?2004l\x1b[<1u"


def test_a_pop_by_the_program_is_counted() -> None:
    from ai.workspaces.experimental.tty import _TerminalModes

    modes = _TerminalModes()
    modes.feed(b"\x1b[>1u\x1b[>1u\x1b[<u")
    assert (
        modes.restore() == b"\x1b[<1u"
    ), "two pushed, one popped: one left to pop"


MODES_DRIVER = """
import asyncio, sys
from ai.workspaces.experimental import tty
from ai.workspaces.experimental import Local
async def main():
    async with Local(sys.argv[1]) as ws:
        p = await ws.pty(
            [
                "sh",
                "-c",
                "printf '\\\\033[?25l\\\\033[?2004h\\\\033[>1uMODES-ON'; "
                "sleep 30",
            ],
            name=sys.argv[2],
        )
        status = await tty.bridge(p)
        print(f"BRIDGE-RETURNED:{status}", flush=True)
        again = await ws.attach(sys.argv[2])
        await again.close()
asyncio.run(main())
"""


@pytest.mark.timeout(60)
def test_detaching_hands_back_a_terminal_without_the_programs_modes(
    tmp_path: Path,
) -> None:
    """The program is still running after Ctrl-], so it never restores what it
    set; the bridge does, before it returns the terminal to the shell."""
    name = f"bridge-{uuid.uuid4().hex[:8]}"
    pid, master = pty.fork()
    if pid == 0:
        os.chdir(tmp_path)
        os.execv(
            sys.executable,
            [sys.executable, "-c", MODES_DRIVER, str(tmp_path), name],
        )
    try:
        assert b"MODES-ON" in _read_until(master, b"MODES-ON")
        os.write(master, b"\x1d")
        out = _read_until(master, b"BRIDGE-RETURNED:")
        after = out.split(b"MODES-ON", 1)[-1]
        for undo in (b"\x1b[?25h", b"\x1b[?2004l", b"\x1b[<1u"):
            assert undo in after, f"{undo!r} not written on detach"
        assert b"BRIDGE-RETURNED:None" in out
        assert _wait_exit(pid) == 0
    finally:
        with contextlib.suppress(OSError):
            os.close(master)
