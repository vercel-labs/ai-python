"""Pieces of the claude adapter that need no CLI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

from ai.harnesses.experimental._adapters import claude

from . import fakes

DEAD_PID = 2**22 - 1


def _started(pid: int) -> str:
    out = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True,
        text=True,
        env={**os.environ, "TZ": "UTC", "LC_ALL": "C"},
        check=True,
    )
    return out.stdout.strip()


async def test_running_cli_sessions_are_found_in_one_command(
    tmp_path: Path,
) -> None:
    """H12: one command, however many records the CLI left behind; a record
    whose process is gone, or whose pid now names a later process, is not a
    holder."""
    config = tmp_path / "claude"
    (config / "sessions").mkdir(parents=True)
    me = os.getpid()
    records = {
        "live": {"pid": me, "sessionId": "live", "procStart": _started(me)},
        "dead": {"pid": DEAD_PID, "sessionId": "dead", "procStart": "x"},
        "reused": {"pid": me, "sessionId": "reused", "procStart": "y"},
    }
    for name, record in records.items():
        (config / "sessions" / f"{name}.json").write_text(json.dumps(record))
    (config / "sessions" / "junk.json").write_text("{not json")

    async with fakes.LockWorkspace(tmp_path) as ws:
        adapter = claude.ClaudeAdapter()
        adapter._workspace = ws
        adapter._config_dir = str(config)
        ws.execs.clear()
        assert await adapter._cli_sessions() == {"live": me}
        assert len(ws.execs) + len(ws.reads) == 1, (ws.execs, ws.reads)


async def test_no_session_records_is_nobody_running(tmp_path: Path) -> None:
    async with fakes.LockWorkspace(tmp_path) as ws:
        adapter = claude.ClaudeAdapter()
        adapter._workspace = ws
        adapter._config_dir = str(tmp_path / "nowhere")
        assert await adapter._cli_sessions() == {}


async def test_listing_stored_sessions_is_a_fixed_number_of_round_trips(
    tmp_path: Path,
) -> None:
    """H12: the SDK used to load every transcript in turn to list them."""
    import uuid

    import claude_agent_sdk

    from ai.harnesses.experimental._adapters import claude_store

    class SlowStore(claude_store.WorkspaceSessionStore):
        # The SDK's own fallback: list, then load each transcript.
        list_session_summaries = (
            claude_agent_sdk.SessionStore.list_session_summaries
        )

    async with fakes.LockWorkspace(tmp_path) as ws:
        key = claude_agent_sdk.project_key_for_directory(ws.path)
        project = tmp_path / "cfg" / "projects" / key
        project.mkdir(parents=True)
        for n in range(6):
            sid = str(uuid.uuid4())
            record = {
                "type": "user",
                "uuid": str(uuid.uuid4()),
                "parentUuid": None,
                "sessionId": sid,
                "cwd": ws.path,
                "timestamp": "2026-10-02T00:00:00Z",
                "message": {"role": "user", "content": f"task number {n}"},
            }
            (project / f"{sid}.jsonl").write_text(json.dumps(record) + "\n")
        store: Any = claude_store.WorkspaceSessionStore
        fast = store(ws, str(tmp_path / "cfg"))
        slow: Any = SlowStore
        slow = slow(ws, str(tmp_path / "cfg"))

        ws.execs.clear()
        ws.reads.clear()
        quick = await claude_agent_sdk.list_sessions_from_store(fast, ws.path)
        assert len(ws.execs) + len(ws.reads) <= 2, (ws.execs, ws.reads)

        def shape(infos: list[Any]) -> list[tuple[Any, ...]]:
            return sorted(
                (i.session_id, i.first_prompt, i.summary) for i in infos
            )

        full = await claude_agent_sdk.list_sessions_from_store(slow, ws.path)
        assert len(quick) == 6
        assert shape(quick) == shape(full)


async def test_the_remote_claude_store_keeps_its_paths_inside() -> None:
    """A store key is names, never a path out of the store."""
    from ai.harnesses.experimental._adapters import claude_store

    store = await claude_store.WorkspaceSessionStore.discover(
        fakes.FakeWorkspace(), "/c"
    )
    keys: list[Any] = [
        {"project_key": "-fake", "session_id": "abc"},
        {"project_key": "-fake", "session_id": "abc", "subpath": "a/b"},
    ]
    assert [store._path(k) for k in keys] == [
        "/c/projects/-fake/abc.jsonl",
        "/c/projects/-fake/abc/a/b.jsonl",
    ]
    bad: list[Any] = [
        {"project_key": "-fake", "session_id": "../../etc/x"},
        {"project_key": "-fake", "session_id": "/etc/x"},
        {"project_key": "..", "session_id": "abc"},
        {"project_key": "-fake", "session_id": "abc", "subpath": "../x"},
    ]
    for key in bad:
        with pytest.raises(ValueError, match="not a store key"):
            store._path(key)
