"""stop and peek on a conversation here, one open in another terminal."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from afk import verbs
from afk.rows import Row

from ai.harnesses.experimental import SessionInfo
from ai.harnesses.experimental.errors import HarnessError
from ai.types.messages import Message, TextPart

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


@pytest.fixture
def other_terminal() -> Any:
    """A process standing in for a CLI in another terminal."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p
    p.kill()
    p.wait()


async def test_stop_ends_the_process_and_says_which(
    other_terminal: subprocess.Popen[bytes],
    capsys: pytest.CaptureFixture[str],
) -> None:
    await verbs.stop(_here(running=True, pid=other_terminal.pid), None)
    assert other_terminal.wait(timeout=10) == -15  # SIGTERM
    assert capsys.readouterr().out == (
        f"afk: stopped 77e1 (claude, pid {other_terminal.pid})\n"
    )


async def test_stop_of_a_process_already_gone_says_so(
    other_terminal: subprocess.Popen[bytes],
    capsys: pytest.CaptureFixture[str],
) -> None:
    other_terminal.kill()
    other_terminal.wait()
    await verbs.stop(_here(running=True, pid=other_terminal.pid), None)
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

    monkeypatch.setitem(verbs.HARNESSES, "claude-code", harness)
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
