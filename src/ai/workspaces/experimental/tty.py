"""Attach this process's terminal to a workspace pty.

Two ends of one wire: your real terminal here (the tty), and the
pseudo-terminal a program is running on in the workspace (the `Pty`).
`bridge` connects them the way ssh does — raw mode so every keystroke goes
through untouched, window-size changes forwarded, the tty restored exactly
on the way out — and returns when the program ends or you press the detach
key. Detaching leaves a NAMED pty running for a later `Workspace.attach`.

On the way out it also switches off the terminal modes the program switched
on (hidden cursor, bracketed paste, focus reports, the enhanced keyboard
protocol): a program you detach from is still running and never will.

It also returns, with the tty restored and a one-line reason, when anything
on OUR side fails: the terminal read failing, the pty refusing input, or the
pty's output ending because another client attached. Nothing here may fail
silently: a raw-mode terminal has no Ctrl-C, so a bridge that has stopped
working and not returned leaves a person with no way out.

Measured, once, on a real terminal: an earlier bridge read keystrokes through
a pipe transport on a dup of stdin, which set that descriptor non-blocking.
In a shell, stdin, stdout and stderr are ONE open file description, so stdout
went non-blocking too; a long transcript redraw then truncated at the tty's
buffer (the screen showed two lines and went dark), the next write failed,
and the shell was left with a non-blocking terminal. Hence: the loop's own
reader on stdin, no dup, no flag changes that outlive the bridge, and writes
that complete.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import sys
import termios
import tty as _tty
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from . import _pty

DETACH_KEY = b"\x1d"
"""Ctrl-]: the key telnet uses, unlikely to collide with a TUI's bindings."""


_DEC_MODE = re.compile(rb"\x1b\[\?([0-9;]+)([hl])")
_KITTY = re.compile(rb"\x1b\[([<>=])([0-9;]*)u")
_MODIFY_OTHER_KEYS = re.compile(rb"\x1b\[>4;?([0-9]*)m")
_CSI_TAIL = re.compile(rb"\x1b(\[[0-9;?<>=]*)?$")
_ON_BY_DEFAULT = {"7", "25"}  # autowrap, visible cursor


class _TerminalModes:
    """The terminal modes a program has switched on, read from its output.

    So the bridge can switch exactly those off when it lets go.

    A program that exits restores its own modes. One you detach from is still
    running and never will — measured: claude leaves the cursor hidden, focus
    reporting, bracketed paste and colour-scheme reports on, and asks for the
    enhanced keyboard protocol, which turns every key into escape codes.
    Only what the program set is undone, so the shell's own modes are left
    as they were.
    """

    def __init__(self) -> None:
        self._dec: dict[str, bool] = {}
        self._kitty_pushed = 0
        self._kitty_set = False
        self._modify_other_keys = False
        self._carry = b""

    def feed(self, data: bytes) -> None:
        data = self._carry + data
        # A sequence split across two reads is completed by the next one.
        tail = _CSI_TAIL.search(data[-24:])
        cut = len(data) - len(tail.group(0)) if tail else len(data)
        data, self._carry = data[:cut], data[cut:]
        for m in _DEC_MODE.finditer(data):
            for mode in m.group(1).decode().split(";"):
                self._dec[mode] = m.group(2) == b"h"
        for m in _KITTY.finditer(data):
            kind, arg = m.group(1), m.group(2).split(b";")[0]
            if kind == b">":
                self._kitty_pushed += 1
            elif kind == b"<":
                self._kitty_pushed = max(
                    0, self._kitty_pushed - (int(arg) if arg else 1)
                )
            else:
                self._kitty_set = True
        for m in _MODIFY_OTHER_KEYS.finditer(data):
            self._modify_other_keys = m.group(1) not in (b"", b"0")

    def restore(self) -> bytes:
        """Return the bytes that put back what the program changed."""
        out = b""
        for mode, on in self._dec.items():
            default = mode in _ON_BY_DEFAULT
            if on != default:
                out += f"\x1b[?{mode}{'h' if default else 'l'}".encode()
        if self._kitty_pushed:
            out += f"\x1b[<{self._kitty_pushed}u".encode()
        elif self._kitty_set:
            out += b"\x1b[=0;1u"
        if self._modify_other_keys:
            out += b"\x1b[>4m"
        return out


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte, in order.

    A tty write may be short (a signal such as SIGWINCH mid-write) and a redraw
    that lands half is a corrupt screen.
    """
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


async def bridge(
    pty: _pty.Pty, *, detach_key: bytes | None = DETACH_KEY
) -> int | None:
    """Drive `pty` from this process's terminal.

    Until the program ends or the detach key is pressed. Returns the exit
    status, or None on detach.
    """
    fd_in, fd_out = sys.stdin.fileno(), sys.stdout.fileno()
    if not os.isatty(fd_in):
        raise RuntimeError("tty.bridge needs a terminal on stdin")
    saved = termios.tcgetattr(fd_in)
    loop = asyncio.get_running_loop()
    detached = asyncio.Event()
    # The loop keeps only weak references to tasks; these must not be
    # collected mid-send.
    background: set[asyncio.Task[None]] = set()

    def spawn(coro: Coroutine[Any, Any, None]) -> None:
        task = loop.create_task(coro)
        background.add(task)
        task.add_done_callback(background.discard)

    async def size() -> None:
        cols, rows = shutil.get_terminal_size()
        # A program that has already ended cannot be resized; its output and
        # status still come through screen() and wait(), so this is no failure.
        with contextlib.suppress(Exception):
            await pty.resize(cols, rows)

    def on_winch(*_: object) -> None:
        spawn(size())

    ended: asyncio.Future[str] = loop.create_future()
    modes = _TerminalModes()

    def end(reason: str) -> None:
        if not ended.done():
            ended.set_result(reason)

    def on_keys() -> None:
        # The loop's own reader on the tty fd: no dup, so the blocking mode of
        # the shell's stdin is never touched, and no pipe transport that can
        # fail out of sight. Every failure ENDS the bridge — a raw-mode
        # terminal whose reader has died has no Ctrl-C and no way out.
        try:
            data = os.read(fd_in, 4096)
        except (OSError, ValueError) as exc:
            end(f"terminal read failed: {exc}")
            return
        if not data:
            end("terminal closed")
            return
        if detach_key and detach_key in data:
            before = data.split(detach_key, 1)[0]
            if before:
                spawn(pty.send(before))
            detached.set()
            return
        spawn(_send(data))

    async def _send(data: bytes) -> None:
        try:
            await pty.send(data)
        except Exception as exc:
            end(f"the pty stopped taking input: {exc}")

    async def screen() -> str:
        while True:
            data = await pty.receive()
            if not data:
                # An exit frame settles wait() before the output ends; if it
                # did not come, the connection ended with the program alive.
                if await pty.wait() is None:
                    return (
                        "the connection ended: another client attached, or "
                        "the link dropped"
                    )
                return ""
            modes.feed(data)
            _write_all(fd_out, data)

    # TCSADRAIN, not setraw's default TCSAFLUSH: flushing discards anything
    # typed before the bridge was ready, and a human's keystrokes are not
    # ours to drop.
    _tty.setraw(fd_in, termios.TCSADRAIN)
    # A blocking terminal for the bridge's whole life: a non-blocking tty
    # truncates bursts and fails when its buffer is full, and something
    # before us may have left it that way. Restored with the rest.
    blocking = (os.get_blocking(fd_in), os.get_blocking(fd_out))
    os.set_blocking(fd_in, True)
    os.set_blocking(fd_out, True)
    previous = signal.signal(signal.SIGWINCH, on_winch)
    loop.add_reader(fd_in, on_keys)
    reason = ""
    try:
        await size()
        screen_task = asyncio.create_task(screen())
        detach_task = asyncio.create_task(detached.wait())
        waiters: set[asyncio.Future[Any]] = {screen_task, detach_task, ended}
        done, _ = await asyncio.wait(
            waiters,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if screen_task in done:
            reason = screen_task.result()
        elif ended in done:
            reason = ended.result()
        for task in (screen_task, detach_task):
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        if detached.is_set():
            await pty.detach()
            return None
        if ended.done():
            # Our side failed, not the program: let go without ending it.
            with contextlib.suppress(Exception):
                await pty.detach()
            return None
        return await pty.wait()
    finally:
        loop.remove_reader(fd_in)
        signal.signal(signal.SIGWINCH, previous)
        # Undo the modes the program switched on — a program left running
        # (detached, displaced) never will, and the shell would inherit them.
        _write_all(fd_out, modes.restore())
        termios.tcsetattr(fd_in, termios.TCSADRAIN, saved)
        _write_all(fd_out, b"\r\n")
        if reason:
            _write_all(fd_out, f"[tty.bridge] {reason}\r\n".encode())
        os.set_blocking(fd_in, blocking[0])
        os.set_blocking(fd_out, blocking[1])
