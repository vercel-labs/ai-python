"""The list against real harnesses: nothing registered, everything found."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from afk import state as st
from afk.rows import local_rows, remote_rows

from ai.harnesses.experimental import Handle, claude_code
from ai.workspaces.experimental import Local, WorkspaceCoords

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.live


async def test_a_conversation_started_outside_afk_is_listed(
    tmp_path: Path,
) -> None:
    """Start one the way a user would, through the harness; afk sees it."""
    async with Local(tmp_path) as ws:
        async with claude_code(workspace=ws) as agent:
            session = agent.session()
            await session.run("Reply with exactly: OK")
            sid = session.session_id
    rows, notes, _ = await local_rows(tmp_path, None)
    assert sid in {r.session_id for r in rows}, notes
    row = next(r for r in rows if r.session_id == sid)
    assert row.where == "here" and row.status.startswith(
        ("active", "idle", "in use")
    )


async def test_the_list_forgets_only_what_the_platform_says_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale record (a sandbox that no longer exists) is forgotten; a record
    afk merely cannot check right now is kept, or a running machine would be
    lost to it for good."""
    monkeypatch.setenv("AFK_STATE", str(tmp_path / "state.json"))
    sandbox = f"ai-no-such-{uuid.uuid4().hex[:10]}"
    handle = Handle(
        kind="claude-code",
        session_id=str(uuid.uuid4()),
        workspace=WorkspaceCoords(provider="vercel-sandbox", location=sandbox),
    )
    record = st.Remote(
        label="stale",
        origin=str(tmp_path),
        sandbox=sandbox,
        pty="afk-stale",
        mode="tui",
        handle=handle,
        pushed_at=st.now(),
    )
    state = st.State(remotes=[record])
    rows, gone = await remote_rows(state, tmp_path, None)
    assert gone == [record.sandbox] and rows[0].status == "gone"

    # Now the platform cannot be asked at all: nothing is "gone". (From a
    # directory with no .env.local above it, which the SDK would otherwise
    # fall back to for credentials.)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
    monkeypatch.setenv("VERCEL_TOKEN", "not-a-real-token")
    monkeypatch.setenv("VERCEL_TEAM_ID", "team_x")
    monkeypatch.setenv("VERCEL_PROJECT_ID", "prj_x")
    rows, gone = await remote_rows(state, tmp_path, None)
    assert gone == [], "an unreachable sandbox must not be forgotten"
    assert rows[0].status == "unreachable"
