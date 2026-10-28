"""Pump bytes between stdio and a pty holder's socket.

Runs INSIDE the workspace, on the platform's interactive PTY (stdlib only).
It does not understand the frames it carries: the SDK on the far side and
the holder on this side speak the protocol; this is just the wire between
them when the workspace is a remote machine. Its one piece of intelligence
is the readiness marker, printed once connected so the SDK knows bytes sent
from here on reach the socket and are not swallowed by a shell still
starting up. The marker arrives as an ARGUMENT, never a literal in this
source, so a traceback echoing this file cannot fake readiness.

Usage: _pty_client.py SOCKET MARKER [WAIT]

WAIT is how many seconds to wait for the socket to appear — a pty just
started has a holder still binding it. Waiting here, beside the socket, is
free; waiting from the SDK meant a network round trip per look.
"""

import os
import select
import socket
import sys
import time


def connect(path: str, wait: float) -> socket.socket:
    deadline = time.monotonic() + wait
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            if time.monotonic() >= deadline:
                sys.stderr.write(f"no pty holder is listening on {path}\n")
                sys.exit(1)
            time.sleep(0.02)


def main() -> None:
    sock = connect(
        sys.argv[1], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    )
    sys.stdout.write(sys.argv[2] + "\n")
    sys.stdout.flush()
    while True:
        ready, _, _ = select.select([0, sock], [], [])
        if 0 in ready:
            data = os.read(0, 65536)
            if not data:
                return
            sock.sendall(data)
        if sock in ready:
            data = sock.recv(65536)
            if not data:
                return
            os.write(1, data)


if __name__ == "__main__":
    main()
