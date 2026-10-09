"""stop and peek on a conversation here, one open in another terminal."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from afk import rows, verbs
from afk.rows import Row

from ai.harnesses.experimental import SessionInfo
from ai.harnesses.experimental.errors import HarnessError
from ai.types.messages import Message, TextPart, ToolCallPart

if TYPE_CHECKING:
    from pathlib import Path


def _here(**kw: Any) -> Row:
    return Row(
        where="here",
        kind="claude-code",
        session_id="77e1ffff",
        status="in use 1m",
        **kw,
    )


def _detached(code: str) -> int:
    """A process standing in for a CLI in another terminal: not afk's child,
    so nothing here reaps it, as nothing in afk reaps a real one."""
    out = subprocess.run(
        [
            "sh",
            "-c",
            f'"{sys.executable}" -c "{code}" >/dev/null 2>&1 & echo $!',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout)


@pytest.fixture
def other_terminal() -> Any:
    pids: list[int] = []

    def spawn(code: str = "import time; time.sleep(60)") -> int:
        pids.append(_detached(code))
        return pids[-1]

    yield spawn
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


async def test_stop_returns_once_the_process_is_gone(
    other_terminal: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    pid = other_terminal()
    await verbs.stop(_here(running=True, pid=pid), None)
    # Gone by the time stop returns: a take-over's resume never overlaps it.
    assert not verbs._alive(pid)
    assert capsys.readouterr().out == (
        f"afk: stopped 77e1 (claude, pid {pid})\n"
    )


async def test_stop_says_when_the_process_will_not_exit(
    other_terminal: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = other_terminal(
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    await asyncio.sleep(0.5)  # let it install the handler
    monkeypatch.setattr(verbs, "STOP_WAIT", 0.5)
    with pytest.raises(HarnessError, match=r"did not exit within 0\.5s"):
        await verbs.stop(_here(running=True, pid=pid), None)
    assert verbs._alive(pid)


async def test_stop_of_a_process_already_gone_says_so(
    other_terminal: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    pid = other_terminal()
    os.kill(pid, signal.SIGKILL)
    while verbs._alive(pid):
        await asyncio.sleep(0.05)
    await verbs.stop(_here(running=True, pid=pid), None)
    assert capsys.readouterr().out == "afk: 77e1 had already exited\n"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_here(), "77e1 is not open anywhere here"),
        (_here(running=True), "afk cannot tell which process has 77e1 open"),
    ],
)
async def test_stop_without_a_process_ends_nothing(
    row: Row, message: str
) -> None:
    with pytest.raises(HarnessError, match=message):
        await verbs.stop(row, None)


class _Agent:
    """The harness peek opens here: a stored transcript, and whether the
    conversation is still open."""

    def __init__(self, messages: list[Message], *, running: bool) -> None:
        self.messages = messages
        self.running = running

    async def __aenter__(self) -> _Agent:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    async def history(self, session_id: str) -> list[Message]:
        return self.messages

    async def sessions(self) -> list[SessionInfo]:
        return [
            SessionInfo(
                kind="claude-code",
                session_id="77e1ffff",
                running=self.running,
            )
        ]


async def test_peek_here_reads_this_directory_and_offers_no_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[Any] = []
    agent = _Agent(
        [
            Message(role="user", parts=[TextPart(text="fix the router")]),
            Message(role="assistant", parts=[TextPart(text="On it.")]),
        ],
        running=False,
    )

    def harness(*, workspace: Any) -> _Agent:
        opened.append(workspace)
        return agent

    monkeypatch.setitem(rows.HARNESSES, "claude-code", harness)
    await verbs.peek(_here(), None, cwd=tmp_path, every=0)
    assert opened[0].kind == "local"
    assert str(opened[0].path) == str(tmp_path.resolve())
    out = capsys.readouterr().out
    assert "watching 77e1 · claude · here" in out
    assert "fix the router" in out and "On it." in out
    assert out.rstrip().endswith("afk: 77e1 is not running")
    assert "afk pull" not in out


async def test_peek_needs_somewhere_to_look() -> None:
    with pytest.raises(HarnessError, match="nothing to peek at"):
        await verbs.peek(_here(), None)


def _peek_at(
    monkeypatch: pytest.MonkeyPatch, messages: list[Message], *, running: bool
) -> None:
    agent = _Agent(messages, running=running)
    monkeypatch.setitem(
        rows.HARNESSES, "claude-code", lambda *, workspace: agent
    )


INTERRUPTED = [
    Message(role="user", parts=[TextPart(text="fix the router")]),
    Message(
        role="user", parts=[TextPart(text="[Request interrupted by user]")]
    ),
]


async def test_peek_stops_when_a_session_open_here_goes_quiet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Still open, but its transcript ends on an interrupted turn, which never
    # reads as a finished one: peek used to follow it forever.
    _peek_at(monkeypatch, INTERRUPTED, running=True)
    monkeypatch.setattr(verbs, "PEEK_IDLE", 0.3)
    await asyncio.wait_for(
        verbs.peek(_here(), None, cwd=tmp_path, every=0.05), 5
    )
    assert (
        capsys.readouterr().out.rstrip().endswith("it may be waiting for input")
    )


async def test_peek_keeps_following_while_a_tool_call_is_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A long tool call (a test suite) is quiet too, and not idle.
    call = ToolCallPart(tool_call_id="c1", tool_name="Bash", tool_args="{}")
    _peek_at(
        monkeypatch,
        [INTERRUPTED[0], Message(role="assistant", parts=[call])],
        running=True,
    )
    monkeypatch.setattr(verbs, "PEEK_IDLE", 0.1)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            verbs.peek(_here(), None, cwd=tmp_path, every=0.05), 1
        )
    # Stopping the watch says the agent goes on.
    assert (
        capsys.readouterr()
        .out.rstrip()
        .endswith("afk: 77e1 keeps going; you stopped watching")
    )


async def test_peek_on_a_terminal_says_it_is_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _peek_at(monkeypatch, INTERRUPTED, running=True)
    monkeypatch.setattr(verbs, "PEEK_IDLE", 0.6)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    await asyncio.wait_for(
        verbs.peek(_here(), None, cwd=tmp_path, every=0.1), 5
    )
    out = capsys.readouterr().out
    assert "following 77e1 · nothing new for 0s · Ctrl-C to stop" in out
    # Redrawn in place, and cleared before the line that ends it.
    assert "\r\x1b[2K" in out
    assert out.rstrip().endswith("it may be waiting for input\x1b[0m")


async def test_peek_piped_draws_no_status_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _peek_at(monkeypatch, INTERRUPTED, running=True)
    monkeypatch.setattr(verbs, "PEEK_IDLE", 0.3)
    await asyncio.wait_for(
        verbs.peek(_here(), None, cwd=tmp_path, every=0.05), 5
    )
    out = capsys.readouterr().out
    assert "following" not in out and "\x1b" not in out
