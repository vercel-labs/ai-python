#!/usr/bin/env python3
"""End-to-end smoke test for the multi-agent textual demo.

Launches the FastAPI server, opens the Textual client in a detached
tmux session, types "y" + Enter twice (one approval per sub-agent),
waits for completion, and asserts the summary panel rendered.

Requires AI_GATEWAY_API_KEY in the environment.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from typing import cast

import ai

HERE = Path(__file__).resolve().parent
SESSION = f"multiagent-e2e-{os.getpid()}"
SERVER_PORT = os.environ.get("SERVER_PORT", "8000")
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"


def _check_health() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/api/health", timeout=1) as r:
            return cast(HTTPResponse, r).status == 200
    except Exception:
        return False


def _capture_pane() -> str:
    return subprocess.check_output(
        ["tmux", "capture-pane", "-t", SESSION, "-p"], text=True
    )


def _wait_for_pane(needle: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pane = _capture_pane()
        except subprocess.CalledProcessError:
            time.sleep(0.5)
            continue
        if needle in pane:
            return
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {needle!r}\n{_capture_pane()}")


def _send_keys(*keys: str) -> None:
    subprocess.run(["tmux", "send-keys", "-t", SESSION, *keys], check=True)


def _extract_panel(pane: str, title: str) -> str:
    """Return the content of the panel whose top border carries *title*.

    Panels are bounded by box-drawing borders. The top border looks like
    ``┌─  summary  ──...──┐`` and the bottom like ``└──...── complete ─┘``.
    Side borders are ``│`` on each line in between.
    """
    lines = pane.splitlines()
    top_re = re.compile(r"┌.*\b" + re.escape(title) + r"\b.*┐$")
    body: list[str] = []
    in_panel = False
    for line in lines:
        if not in_panel and top_re.search(line):
            in_panel = True
            continue
        if in_panel and line.startswith("└"):
            break
        if in_panel:
            body.append(line.strip("│").strip())
    return "\n".join(body).strip()


def main() -> int:
    if not ai.get_provider("vercel").is_configured():
        print("AI Gateway provider is not configured", file=sys.stderr)
        return 1
    if not shutil.which("tmux"):
        print("tmux not found in PATH", file=sys.stderr)
        return 1

    logs = Path(tempfile.mkdtemp())
    print(f"Logs: {logs}")

    server: subprocess.Popen[bytes] | None = None
    server_log = (logs / "server.log").open("wb")

    def cleanup() -> None:
        subprocess.run(
            ["tmux", "kill-session", "-t", SESSION],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
        server_log.close()

    try:
        print(f"Starting server on :{SERVER_PORT}...")
        server = subprocess.Popen(
            [
                "uv",
                "run",
                "--frozen",
                "--with-editable",
                str(Path.home() / "src/py-ai"),
                "fastapi",
                "dev",
                "server.py",
                "--port",
                SERVER_PORT,
            ],
            cwd=HERE,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

        print("Waiting for server...")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if server.poll() is not None:
                print(f"Server died — see {logs}/server.log", file=sys.stderr)
                return 1
            if _check_health():
                break
            time.sleep(0.5)
        else:
            print("Server didn't come up in 30s", file=sys.stderr)
            return 1

        print(f"Starting Textual client in tmux session {SESSION}...")
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                SESSION,
                "-x",
                "80",
                "-y",
                "120",
                (
                    f"cd '{HERE}' && "
                    "uv run --frozen --with-editable ~/src/py-ai/ "
                    "python client.py"
                ),
            ],
            check=True,
        )

        print("Waiting for first approval prompt...")
        _wait_for_pane("approve ")

        print("Sending y + Enter (approval 1)...")
        _send_keys("y", "Enter")
        time.sleep(3)  # let server-side resolution propagate

        print("Waiting for second approval prompt...")
        _wait_for_pane("approve ")

        print("Sending y + Enter (approval 2)...")
        _send_keys("y", "Enter")
        time.sleep(1)

        print("Waiting for run completion...")
        _wait_for_pane("press q to quit", timeout=120)

        pane = _capture_pane()
        print("=== PANE ===")
        print(pane)
        print("=== END ===")

        # Quit the client cleanly.
        _send_keys("q")

        if "awaiting approval" in pane:
            print("FAIL: approval still pending", file=sys.stderr)
            return 1

        summary = _extract_panel(pane, "summary")
        if not summary:
            print(
                "FAIL: summary panel is empty — streaming events routed to "
                "the wrong panel (see TODO in client.py)",
                file=sys.stderr,
            )
            return 1

        print("PASS")
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
