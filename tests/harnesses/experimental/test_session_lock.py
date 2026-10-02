"""The workspace lock, on a real local workspace: atomic, stale-aware,
releasable."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ai.harnesses.experimental._session_lock import STALE_AFTER, SessionLock
from ai.harnesses.experimental.errors import SessionBusyError
from ai.workspaces.experimental import Local

KIND = "lock-test"


@pytest.fixture
async def lock(tmp_path: Path) -> AsyncIterator[SessionLock]:
    async with Local(tmp_path) as ws:
        yield SessionLock(ws, KIND)


async def test_a_live_holder_blocks_a_second_acquire_until_release(
    lock: SessionLock,
) -> None:
    sid = str(uuid.uuid4())
    await lock.acquire(sid, pid=os.getpid())
    try:
        with pytest.raises(SessionBusyError) as info:
            await lock.acquire(sid, pid=os.getpid())
        assert info.value.holder_pid == os.getpid()
        assert (await lock.holders())[sid] == os.getpid()
    finally:
        await lock.release(sid)
    assert await lock.holder(sid) is None
    await lock.acquire(sid, pid=os.getpid())  # free again
    await lock.release(sid)


async def test_a_lock_whose_process_is_gone_is_reclaimed(
    lock: SessionLock,
) -> None:
    sid = str(uuid.uuid4())
    await lock.acquire(sid, pid=2**22 - 1)  # no such process
    try:
        assert await lock.holder(sid) is None, "a dead pid holds nothing"
        await lock.acquire(sid, pid=os.getpid())  # reclaimed, not refused
        holder = await lock.holder(sid)
        assert holder is not None and holder.pid == os.getpid()
    finally:
        await lock.release(sid)


async def test_a_pidless_lock_is_trusted_briefly_then_treated_as_abandoned(
    lock: SessionLock,
) -> None:
    sid = str(uuid.uuid4())
    await lock.acquire(sid)  # a launch in progress: no pid yet
    try:
        with pytest.raises(SessionBusyError):
            await lock.acquire(sid)
        # Age it past the horizon and it is an abandoned launch.
        path = Path(await lock.root()) / sid / "holder.json"
        path.write_text(
            json.dumps(
                {
                    "pid": None,
                    "since": int(time.time() - STALE_AFTER - 1),
                    "client": "",
                }
            )
        )
        await lock.acquire(sid, pid=os.getpid())
    finally:
        await lock.release(sid)


async def test_release_of_an_unheld_lock_is_harmless(lock: SessionLock) -> None:
    await lock.release(str(uuid.uuid4()))
