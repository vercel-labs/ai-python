"""What a sandbox reference means over time: gone, unreachable, reopened.

A caller that keeps sandbox names (afk does) must be told the difference
between a sandbox that no longer exists and one it merely cannot reach, or
it forgets running machines. And one workspace object closed and opened
again may be a different VM: nothing learned about the first may leak.
"""

from __future__ import annotations

import uuid

import pytest

from ai.workspaces.experimental import VercelSandbox
from ai.workspaces.experimental.errors import WorkspaceError, WorkspaceGoneError
from tests.workspaces.experimental.conftest import sandbox_credentials

pytestmark = [pytest.mark.sandbox, pytest.mark.timeout(300)]


@pytest.fixture
def creds() -> None:
    """The sandbox reads its credentials from the environment; skip
    without them."""
    if sandbox_credentials() is None:
        pytest.skip(
            "needs VERCEL_TOKEN/TEAM_ID/PROJECT_ID or VERCEL_OIDC_TOKEN"
        )


async def test_a_sandbox_that_does_not_exist_is_gone(creds: None) -> None:
    with pytest.raises(WorkspaceGoneError):
        await VercelSandbox(name=f"ai-no-such-{uuid.uuid4().hex[:10]}").open()


async def test_a_stopped_sandbox_is_gone(creds: None) -> None:
    async with VercelSandbox(keep=False) as ws:
        name = ws.name
    assert name
    with pytest.raises(WorkspaceGoneError):
        await VercelSandbox(name=name).open()


async def test_a_sandbox_that_cannot_be_reached_is_not_gone() -> None:
    """Bad credentials stand in for any failure to ask: the platform never
    answered about the sandbox, so nobody may conclude it is gone."""
    with pytest.raises(WorkspaceError) as caught:
        await VercelSandbox(
            name="whatever",
            token="not-a-real-token",
            team_id="team_x",
            project_id="prj_x",
        ).open()
    assert not isinstance(caught.value, WorkspaceGoneError)


async def test_a_reopened_workspace_learns_its_new_vm_afresh(
    creds: None,
) -> None:
    """Closing and opening the same object makes a new VM.

    Its home and its pty scripts are its own: a pty started there must work.
    """
    ws = VercelSandbox(keep=False)
    await ws.open()
    first = ws.name
    p = await ws.pty(["sh", "-c", "echo first-vm"])
    await p.close()
    await ws.close()
    await ws.open()
    try:
        assert ws.name and ws.name != first, "no name means a fresh VM"
        p = await ws.pty(["sh", "-c", "echo second-vm; exit 3"])
        seen = b""
        while chunk := await p.receive():
            seen += chunk
        assert (
            b"second-vm" in seen
        ), "the scripts must be uploaded into the new VM too"
        assert await p.wait() == 3
    finally:
        await ws.close()


async def test_a_transcript_the_sdk_writes_is_dated_now(creds: None) -> None:
    """Files written through the sandbox's file API are stamped 1970, and a
    running claude CLI deletes transcripts it takes for ancient — a fork's
    file once vanished before its CLI could resume it."""
    import time

    from claude_agent_sdk.types import SessionKey

    from ai.harnesses.experimental._adapters.claude_store import (
        WorkspaceSessionStore,
    )

    async with VercelSandbox(keep=False) as ws:
        store = await WorkspaceSessionStore.discover(ws)
        key: SessionKey = {
            "project_key": "-vercel-sandbox",
            "session_id": str(uuid.uuid4()),
        }
        await store.append(key, [{"type": "user", "uuid": str(uuid.uuid4())}])
        path = store._path(key)
        mtime = float(
            (await ws.exec(["stat", "-c", "%Y", path])).stdout.strip()
        )
        assert (
            abs(mtime - time.time()) < 600
        ), f"transcript dated {time.ctime(mtime)}"
