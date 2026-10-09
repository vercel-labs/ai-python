"""A CLI the SDK did not launch is still seen holding its conversation.

On a real local workspace, with real processes standing in for the CLIs: a
claude process is known by the record the CLI writes for it, a codex one by
its command line. A record left behind, or naming a pid a later process
reuses, holds nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from ai.harnesses.experimental import _cli_processes
from ai.harnesses.experimental._adapters.claude import ClaudeAdapter
from ai.harnesses.experimental._adapters.codex import CodexAdapter
from ai.harnesses.experimental._session_lock import SessionLock
from ai.workspaces.experimental import Local


def _started(pid: int) -> str:
    """What claude records as `procStart`: `ps` lstart, in UTC."""
    return subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": "UTC", "LC_ALL": "C"},
        check=True,
    ).stdout.strip()


@pytest.fixture
def spawn() -> Iterator[list[subprocess.Popen[bytes]]]:
    """Processes that stand in for CLIs, ended after the test."""
    procs: list[subprocess.Popen[bytes]] = []
    yield procs
    for p in procs:
        p.kill()
        p.wait()


def _sleeper(argv0: str = "sleeper", *args: str) -> subprocess.Popen[bytes]:
    """A process whose command line reads `argv0 ... args`."""
    return subprocess.Popen(
        [argv0, "-c", "import time; time.sleep(60)", *args],
        executable=sys.executable,
    )


@pytest.fixture
async def ws(tmp_path: Path) -> AsyncIterator[Local]:
    config = tmp_path / "claude-config"
    (config / "sessions").mkdir(parents=True)
    async with Local(tmp_path, env={"CLAUDE_CONFIG_DIR": str(config)}) as ws:
        yield ws


def _claude(ws: Local) -> ClaudeAdapter:
    # What start() sets, without probing for a real `claude`.
    adapter = ClaudeAdapter()
    adapter._workspace = ws
    adapter._config_dir = ws.env["CLAUDE_CONFIG_DIR"]
    # A kind of its own: the lock lives under the real HOME.
    adapter._lock = SessionLock(ws, "cli-processes-test-claude")
    return adapter


def _codex(ws: Local) -> CodexAdapter:
    adapter = CodexAdapter()
    adapter._workspace = ws
    adapter._lock = SessionLock(ws, "cli-processes-test-codex")
    return adapter


def _record(ws: Local, pid: int, session_id: str, started: str) -> None:
    path = Path(ws.env["CLAUDE_CONFIG_DIR"]) / "sessions" / f"{pid}.json"
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "sessionId": session_id,
                "procStart": started,
                "kind": "interactive",
            }
        )
    )


async def test_processes_lists_this_one_with_its_start_and_argv(
    ws: Local, spawn: list[subprocess.Popen[bytes]]
) -> None:
    p = _sleeper("sleeper", "--flag", "value")
    spawn.append(p)
    found = await _cli_processes.processes(ws)
    assert os.getpid() in found
    mine = found[p.pid]
    assert mine.started == _started(p.pid)
    assert mine.argv[0] == "sleeper"
    assert mine.argv[-2:] == ["--flag", "value"]


async def test_a_claude_process_holds_the_conversation_it_records(
    ws: Local, spawn: list[subprocess.Popen[bytes]]
) -> None:
    live = _sleeper()
    spawn.append(live)
    sid = str(uuid.uuid4())
    _record(ws, live.pid, sid, _started(live.pid))
    assert await _claude(ws).running_sessions() == {sid: live.pid}


async def test_a_record_left_behind_holds_nothing(
    ws: Local, spawn: list[subprocess.Popen[bytes]]
) -> None:
    gone = _sleeper()
    started = _started(gone.pid)
    gone.kill()
    gone.wait()
    _record(ws, gone.pid, str(uuid.uuid4()), started)
    assert await _claude(ws).running_sessions() == {}


async def test_a_reused_pid_is_not_the_recorded_process(
    ws: Local, spawn: list[subprocess.Popen[bytes]]
) -> None:
    # Same pid, a different start: a later process, not the claude that
    # wrote the record. Stopping it would end the wrong process.
    live = _sleeper()
    spawn.append(live)
    _record(ws, live.pid, str(uuid.uuid4()), "Thu Jan  1 00:00:00 1970")
    assert await _claude(ws).running_sessions() == {}


async def test_a_record_that_is_not_json_is_skipped(ws: Local) -> None:
    sessions = Path(ws.env["CLAUDE_CONFIG_DIR"]) / "sessions"
    (sessions / "123.json").write_text("{not json")
    (sessions / "456.json").write_text(json.dumps({"pid": 456}))
    assert await _claude(ws).running_sessions() == {}


async def test_no_sessions_directory_is_no_holders(tmp_path: Path) -> None:
    async with Local(
        tmp_path, env={"CLAUDE_CONFIG_DIR": str(tmp_path / "none")}
    ) as ws:
        adapter = _claude(ws)
        assert await adapter.running_sessions() == {}


async def test_the_sdk_lock_and_the_cli_record_merge(
    ws: Local, spawn: list[subprocess.Popen[bytes]]
) -> None:
    adapter = _claude(ws)
    assert adapter._lock is not None
    by_sdk, by_cli = str(uuid.uuid4()), str(uuid.uuid4())
    await adapter._lock.acquire(by_sdk, pid=os.getpid())
    try:
        live = _sleeper()
        spawn.append(live)
        _record(ws, live.pid, by_cli, _started(live.pid))
        assert await adapter.running_sessions() == {
            by_sdk: os.getpid(),
            by_cli: live.pid,
        }
    finally:
        await adapter._lock.release(by_sdk)


async def test_the_cli_pid_wins_over_a_pidless_lock(
    ws: Local, spawn: list[subprocess.Popen[bytes]]
) -> None:
    # A TUI the SDK launched: locked before its pid is known, and the CLI's
    # own record says which process it is.
    adapter = _claude(ws)
    assert adapter._lock is not None
    sid = str(uuid.uuid4())
    await adapter._lock.acquire(sid)
    try:
        live = _sleeper()
        spawn.append(live)
        _record(ws, live.pid, sid, _started(live.pid))
        assert await adapter.running_sessions() == {sid: live.pid}
    finally:
        await adapter._lock.release(sid)


@pytest.mark.parametrize(
    "args",
    [
        ["resume", "{tid}"],
        ["exec", "resume", "{tid}"],
        ["resume", "--model", "gpt", "{tid}"],
    ],
)
async def test_codex_resumed_by_id_holds_its_thread(
    ws: Local, spawn: list[subprocess.Popen[bytes]], args: list[str]
) -> None:
    tid = str(uuid.uuid4())
    p = _sleeper("/usr/local/bin/codex", *(a.format(tid=tid) for a in args))
    spawn.append(p)
    # Only this thread: a real codex on this machine may hold others.
    assert (await _codex(ws).running_sessions()).get(tid) == p.pid


@pytest.mark.parametrize(
    ("argv0", "args"),
    [
        # A fresh codex names no thread; nor does resume --last.
        ("codex", []),
        ("codex", ["resume", "--last"]),
        # The app-server daemon holds threads, but is not one TUI's process.
        ("codex", ["app-server", "--listen", "unix://"]),
        # Not codex at all, though it says resume and an id.
        ("other", ["resume", "{tid}"]),
    ],
)
async def test_codex_processes_that_name_no_thread_hold_nothing(
    ws: Local,
    spawn: list[subprocess.Popen[bytes]],
    argv0: str,
    args: list[str],
) -> None:
    tid = str(uuid.uuid4())
    p = _sleeper(argv0, *(a.format(tid=tid) for a in args))
    spawn.append(p)
    running = await _codex(ws).running_sessions()
    assert tid not in running
    assert p.pid not in running.values()
