"""Hold a program on a pseudo-terminal; let one client attach and detach.

Runs INSIDE the workspace as a standalone script (stdlib only — it cannot
import ai there). It owns the master side of a pty, so the program
never sees its terminal disappear when a client goes away; keeps a bounded
scrollback so a reattaching client sees where things stand; and serves one
client at a time over a Unix socket. It lives exactly as long as the
program and removes its socket and pid file when the program ends — the pty
analogue of the FIFO supervisor in `_sandbox.py`.

Frames, both directions: 1 byte type, 4 bytes big-endian length, payload.
    client -> holder:  i input   r resize "cols,rows"   d detach   c close
    holder -> client:  o output  x exit "status"
`ai.workspaces.experimental._pty` speaks the same protocol from the SDK side.

Usage:
    _pty_holder.py SOCKET [--transient] [--cols N] [--rows N] -- PROGRAM [ARGS]
    --transient  end the program when the client detaches (an unnamed pty)
"""

import contextlib
import fcntl
import os
import pty
import select
import signal
import socket
import struct
import sys
import termios
import time
from collections.abc import Iterator

FIRST_CLIENT_GRACE = 20.0
"""Seconds a holder whose program ended before any client connected waits
for one, so a short program's output and status still reach whoever started it.
Long enough for a sandbox client: polling for the socket, then opening the
interactive session that carries the frames, takes several seconds there."""

SCROLLBACK = 256 * 1024


def frame(kind: bytes, payload: bytes = b"") -> bytes:
    return kind + struct.pack(">I", len(payload)) + payload


class Parser:
    def __init__(self) -> None:
        self.buf = b""

    def feed(self, data: bytes) -> Iterator[tuple[bytes, bytes]]:
        self.buf += data
        while len(self.buf) >= 5:
            kind, length = self.buf[:1], struct.unpack(">I", self.buf[1:5])[0]
            if len(self.buf) < 5 + length:
                return
            payload, self.buf = self.buf[5 : 5 + length], self.buf[5 + length :]
            yield kind, payload


def set_size(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _unlink(*paths: str) -> None:
    for p in paths:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(p)


def main() -> int:
    args = sys.argv[1:]
    sock_path = args.pop(0)
    transient = False
    cols, rows = 80, 24
    while args and args[0] != "--":
        flag = args.pop(0)
        if flag == "--transient":
            transient = True
        elif flag == "--cols":
            cols = int(args.pop(0))
        elif flag == "--rows":
            rows = int(args.pop(0))
    if args and args[0] == "--":
        args.pop(0)
    program = args

    # The socket before the program: a client that sees the socket can
    # connect before the program has written a byte, so nothing it prints
    # is ever older than the scrollback. (Sockets are not inherited across
    # exec, so the program never sees this one.)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _unlink(sock_path)
    server.bind(sock_path)
    os.chmod(sock_path, 0o600)
    server.listen(1)

    pid, master = pty.fork()
    if pid == 0:
        os.execvp(program[0], program)
    set_size(master, cols, rows)
    with open(sock_path + ".pid", "w") as f:
        f.write(f"{pid}\n")

    scrollback = b""
    client = None
    parser = Parser()

    def send(data: bytes) -> None:
        nonlocal client
        if client is None:
            return
        try:
            client.sendall(data)
        except OSError:
            drop_client()

    def mark_attached(*, on: bool) -> None:
        # `<sock>.client` exists exactly while a client is connected, so a
        # listing can say "open elsewhere" without connecting — connecting
        # would take the pty from whoever has it.
        if on:
            with open(sock_path + ".client", "w") as f:
                f.write(f"{int(time.time())}\n")
        else:
            _unlink(sock_path + ".client")

    def drop_client() -> None:
        nonlocal client
        if client is not None:
            try:
                client.close()
            finally:
                client = None
        mark_attached(on=False)
        if transient:
            end_program()

    def end_program() -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGHUP)

    def drain_master() -> None:
        # Whatever the program wrote last may still sit in the master when
        # its exit is noticed (Linux reports the exit before the EOF).
        nonlocal scrollback
        os.set_blocking(master, False)
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:
                return
            if not data:
                return
            scrollback = (scrollback + data)[-SCROLLBACK:]
            send(frame(b"o", data))

    def finish(raw_status: int) -> int:
        nonlocal client, parser
        drain_master()
        if client is None:
            # Ended before anyone attached: give the client that started it
            # a moment to arrive for the output and the status, rather than
            # unlinking the socket under its feet.
            server.settimeout(FIRST_CLIENT_GRACE)
            with contextlib.suppress(OSError):
                client, parser = server.accept()[0], Parser()
                if scrollback:
                    send(frame(b"o", scrollback))
        status = os.waitstatus_to_exitcode(raw_status)
        send(frame(b"x", str(status).encode()))
        if client is not None:
            client.close()
        server.close()
        _unlink(sock_path, sock_path + ".pid", sock_path + ".client")
        return status

    while True:
        fds = [master, server] + ([client] if client is not None else [])
        try:
            ready, _, _ = select.select(fds, [], [], 1.0)
        except InterruptedError:
            continue
        for fd in ready:
            if fd is server:
                conn, _ = server.accept()
                if client is not None:
                    with contextlib.suppress(OSError):
                        client.close()
                client, parser = conn, Parser()
                mark_attached(on=True)
                if scrollback:
                    send(frame(b"o", scrollback))
            elif fd == master:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    data = b""
                if not data:
                    _, raw = os.waitpid(pid, 0)
                    return finish(raw)
                scrollback = (scrollback + data)[-SCROLLBACK:]
                send(frame(b"o", data))
            elif client is not None and fd is client:
                try:
                    data = client.recv(65536)
                except OSError:
                    data = b""
                if not data:
                    drop_client()
                    continue
                for kind, payload in parser.feed(data):
                    if kind == b"i":
                        os.write(master, payload)
                    elif kind == b"r":
                        c, r = (int(x) for x in payload.decode().split(","))
                        set_size(master, c, r)
                        os.kill(pid, signal.SIGWINCH)
                    elif kind == b"d":
                        drop_client()
                    elif kind == b"c":
                        end_program()
        # A program that ended while nobody was attached still needs reaping.
        try:
            done, raw = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            done, raw = 0, 0
        if done:
            return finish(raw)


if __name__ == "__main__":
    sys.exit(main() or 0)
