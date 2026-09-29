"""A program on a pseudo-terminal, wherever the workspace is.

`Workspace.pty(argv)` runs a program that believes it is on a terminal —
`isatty` is true, the line discipline applies, TUIs render — and hands back
its master side as a `Pty`: bytes in, bytes out, a window size, and the two
verbs the rest of the SDK uses for letting go. `spawn` is for programs you
talk to over a protocol; `pty` is for programs a human talks to.

A NAME MEANS PERSISTENCE, as it does for a sandbox. An unnamed pty ends with
your connection. A named one is held open by a holder in the workspace
(`_pty_holder.py`) that lives exactly as long as the program, keeps a
bounded scrollback, and lets a later `Workspace.attach(name)` — from any
process — pick up where you left off. `detach()` drops your end and leaves a
named program running; `close()` ends it.

Both workspaces implement this with the same holder and the same frames:
locally over a Unix socket, remotely over the platform's interactive PTY
with `_pty_client.py` as the wire. This module is the SDK side of those
frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Frame protocol shared with _pty_holder.py — kept identical there by hand,
# since that file runs where ai cannot be imported.
INPUT, RESIZE, DETACH, CLOSE, OUTPUT, EXIT = b"i", b"r", b"d", b"c", b"o", b"x"

DEFAULT_SIZE = (80, 24)


def frame(kind: bytes, payload: bytes = b"") -> bytes:
    return kind + struct.pack(">I", len(payload)) + payload


class PtyInfo(BaseModel):
    """A named pty still running in a workspace, as `Workspace.ptys()` lists it.

    Learned without connecting: a pty serves one client, and a new
    connection takes it from whoever has it, so a listing must never be one.
    """

    name: str
    """What `Workspace.attach(name)` takes."""

    pid: int
    """The program's pid where it runs."""

    attached: bool
    """Whether a client is connected right now — someone's terminal is on it.

    Attaching anyway takes it over; the displaced client is told so.
    """


class Pty(ABC):
    """The master side of a pseudo-terminal a program is running on."""

    name: str | None
    """Set for a persistent pty: what `Workspace.attach(name)` takes."""

    pid: int | None
    """The program's pid where it RUNS (the holder publishes it), for a
    liveness check that outlives any one client."""

    @abstractmethod
    async def send(self, data: bytes) -> None:
        """Bytes typed at the program's terminal."""

    @abstractmethod
    async def receive(self) -> bytes:
        """Return the next bytes the program wrote to its terminal.

        b"" when it has ended.
        """

    @abstractmethod
    async def resize(self, cols: int, rows: int) -> None:
        """Tell the program its terminal changed size (it gets SIGWINCH)."""

    @abstractmethod
    async def detach(self) -> None:
        """Drop this connection.

        A named program keeps running; an unnamed one ends.
        """

    @abstractmethod
    async def close(self) -> None:
        """End the program, then drop the connection."""

    @abstractmethod
    async def wait(self) -> int | None:
        """Wait for the program's exit status."""


class FramedPty(Pty):
    """A `Pty` speaking the holder's frames over any byte channel.

    The channel is two callables — `send_bytes`, `recv_bytes` — plus a
    `close_channel`, so the same class serves a local Unix socket and a
    remote interactive PTY. Output frames are queued as they arrive; an exit
    frame settles `wait()` and ends `receive()` with b"".
    """

    def __init__(
        self,
        *,
        name: str | None,
        send_bytes: Callable[[bytes], Awaitable[None]],
        recv_bytes: Callable[[], Awaitable[bytes]],
        close_channel: Callable[[], Awaitable[None]],
        pid: int | None = None,
    ) -> None:
        self.name = name
        self.pid = pid
        self._send = send_bytes
        self._recv = recv_bytes
        self._close_channel = close_channel
        self._buf = b""
        self._out: asyncio.Queue[bytes] = asyncio.Queue()
        self._exit: asyncio.Future[int | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._pump = asyncio.create_task(
            self._read_loop(), name="harness-pty-pump"
        )
        self._released = False

    async def _read_loop(self) -> None:
        try:
            while True:
                data = await self._recv()
                if not data:
                    break
                self._buf += data
                while len(self._buf) >= 5:
                    kind, length = (
                        self._buf[:1],
                        struct.unpack(">I", self._buf[1:5])[0],
                    )
                    if len(self._buf) < 5 + length:
                        break
                    payload, self._buf = (
                        self._buf[5 : 5 + length],
                        self._buf[5 + length :],
                    )
                    if kind == OUTPUT:
                        self._out.put_nowait(payload)
                    elif kind == EXIT:
                        status = (
                            int(payload) if payload.strip().isdigit() else None
                        )
                        if not self._exit.done():
                            self._exit.set_result(status)
                        return
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            if not self._exit.done():
                # The channel ended without an exit frame: detached, or the
                # far side vanished. Not an exit status we can vouch for.
                self._exit.set_result(None)
            self._out.put_nowait(b"")

    async def send(self, data: bytes) -> None:
        await self._send(frame(INPUT, data))

    async def receive(self) -> bytes:
        return await self._out.get()

    async def resize(self, cols: int, rows: int) -> None:
        await self._send(frame(RESIZE, f"{cols},{rows}".encode()))

    async def detach(self) -> None:
        if self._released:
            return
        self._released = True
        with contextlib.suppress(Exception):
            await self._send(frame(DETACH))
        await self._release()

    async def close(self) -> None:
        if self._released:
            return
        try:
            await self._send(frame(CLOSE))
            # Give the holder a moment to report the exit status.
            async with asyncio.timeout(10):
                await asyncio.shield(self._exit)
        except Exception:
            pass
        self._released = True
        await self._release()

    async def wait(self) -> int | None:
        return await asyncio.shield(self._exit)

    async def _release(self) -> None:
        self._pump.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._pump
        with contextlib.suppress(Exception):
            async with asyncio.timeout(10):
                await self._close_channel()
