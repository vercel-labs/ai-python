"""The workspace lock, on a real local workspace: atomic, stale-aware,
releasable only by its holder.

Each test gets a workspace whose HOME is a temp dir, so lock files never
land in the user's own home.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ai.harnesses.experimental._session_lock import STALE_AFTER, SessionLock
from ai.harnesses.experimental.errors import SessionBusyError

from . import fakes

KIND = "lock-test"
DEAD_PID = 2**22 - 1  # no such process


@pytest.fixture
async def workspace(tmp_path: Path) -> AsyncIterator[fakes.LockWorkspace]:
    async with fakes.LockWorkspace(tmp_path) as ws:
        yield ws


@pytest.fixture
def lock(workspace: fakes.LockWorkspace) -> SessionLock:
    return SessionLock(workspace, KIND)


async def _record(lock: SessionLock, sid: str) -> dict[str, object]:
    path = Path(await lock.root()) / sid / "holder.json"
    return dict(json.loads(path.read_text()))


async def _plant(lock: SessionLock, sid: str, **record: object) -> None:
    """A lock some other client left behind."""
    directory = Path(await lock.root()) / sid
    directory.mkdir(parents=True)
    (directory / "holder.json").write_text(
        json.dumps({"since": int(time.time()), "client": "", **record})
    )


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
    await lock.acquire(sid, pid=DEAD_PID)
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
            await SessionLock(lock._workspace, KIND).acquire(sid)
        # Age it past the horizon and it is an abandoned launch.
        path = Path(await lock.root()) / sid / "holder.json"
        record = json.loads(path.read_text())
        record["since"] = int(time.time() - STALE_AFTER - 1)
        path.write_text(json.dumps(record))
        await SessionLock(lock._workspace, KIND).acquire(sid, pid=os.getpid())
    finally:
        await lock.release(sid)


async def test_release_of_an_unheld_lock_is_harmless(lock: SessionLock) -> None:
    await lock.release(str(uuid.uuid4()))


async def test_racing_reclaimers_of_a_stale_lock_never_both_win(
    workspace: fakes.LockWorkspace,
) -> None:
    """H6: both saw it stale; the old code let the second delete the first's
    fresh lock and take it too. The interleaving is forced: every removal
    waits until both clients have judged the lock stale, and the second
    removal comes after the first client has finished taking it."""
    arrived: list[asyncio.Event] = []
    both = asyncio.Event()

    async def stagger(argv: list[str]) -> None:
        if argv[0] != "rm":
            return
        mine = asyncio.Event()
        arrived.append(mine)
        if len(arrived) == 2:
            both.set()
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(1):
                await both.wait()
        if arrived.index(mine) == 1:
            await asyncio.sleep(0.3)

    workspace.before_exec = stagger
    sid = str(uuid.uuid4())
    await _plant(SessionLock(workspace, KIND), sid, pid=DEAD_PID)
    clients = [SessionLock(workspace, KIND) for _ in range(2)]
    outcomes = await asyncio.gather(
        *(c.acquire(sid, pid=os.getpid()) for c in clients),
        return_exceptions=True,
    )
    assert [o is None for o in outcomes].count(True) == 1, outcomes
    assert all(
        isinstance(o, SessionBusyError) for o in outcomes if o is not None
    )


async def test_many_racing_acquirers_one_winner(
    workspace: fakes.LockWorkspace,
) -> None:
    for _ in range(5):
        sid = str(uuid.uuid4())
        await _plant(SessionLock(workspace, KIND), sid, pid=DEAD_PID)
        clients = [SessionLock(workspace, KIND) for _ in range(6)]
        outcomes = await asyncio.gather(
            *(c.acquire(sid, pid=os.getpid()) for c in clients),
            return_exceptions=True,
        )
        assert [o is None for o in outcomes].count(True) == 1, outcomes


async def test_only_the_holder_can_release(
    workspace: fakes.LockWorkspace,
) -> None:
    sid = str(uuid.uuid4())
    holder, other = SessionLock(workspace, KIND), SessionLock(workspace, KIND)
    await holder.acquire(sid, pid=os.getpid())
    await other.release(sid)
    assert await holder.holder(sid) is not None, "not the other's to release"
    await other.claim_pid(sid, DEAD_PID)
    assert (await _record(holder, sid))["pid"] == os.getpid()
    await holder.release(sid)
    assert await holder.holder(sid) is None


async def test_a_lock_is_never_seen_without_its_record(
    lock: SessionLock,
) -> None:
    sid = str(uuid.uuid4())
    await lock.acquire(sid)
    root = Path(await lock.root())
    left = sorted(p.name for p in root.iterdir() if p.name != ".mutex")
    assert left == [sid], "no temp left over"
    assert (await _record(lock, sid))["pid"] is None
    await lock.release(sid)


async def test_another_users_process_counts_as_alive(lock: SessionLock) -> None:
    """`kill(pid, 0)` on a process we may not signal raises PermissionError:
    it exists. pid 1 always does, and is never ours."""
    if os.getuid() == 0:
        pytest.skip("root may signal anything")
    sid = str(uuid.uuid4())
    await _plant(lock, sid, pid=1)
    holder = await lock.holder(sid)
    assert holder is not None and holder.pid == 1


async def test_a_reused_pid_is_not_the_holder(lock: SessionLock) -> None:
    """The pid is alive, but it is a later process than the one recorded."""
    sid = str(uuid.uuid4())
    await _plant(lock, sid, pid=os.getpid(), started="Thu Jan  1 00:00:00 1970")
    assert await lock.holder(sid) is None
    await lock.acquire(sid, pid=os.getpid())
    started = (await _record(lock, sid))["started"]
    assert isinstance(started, str) and started
    assert await lock.holder(sid) is not None, "its own start time matches"
    await lock.release(sid)


async def test_listing_holders_is_one_command(
    workspace: fakes.LockWorkspace,
) -> None:
    """H12: the listing used to cost several commands per lock."""
    locks = [SessionLock(workspace, KIND) for _ in range(5)]
    for i, held in enumerate(locks):
        await held.acquire(f"s{i}", pid=os.getpid())
    workspace.execs.clear()
    assert len(await locks[0].holders()) == 5
    assert len(workspace.execs) == 1


@pytest.mark.parametrize(
    "bad", ["/victim", "../victim", "a/../../victim", "", ".", "..", "-rf"]
)
async def test_a_session_id_is_a_name_never_a_path(
    lock: SessionLock, workspace: fakes.LockWorkspace, tmp_path: Path, bad: str
) -> None:
    """Security review: an id joined onto the lock root escaped it, and a
    stale "lock" there was deleted with rmtree. Refused before anything
    runs in the workspace."""
    from ai.harnesses.experimental.errors import HarnessError

    ran = len(workspace.execs)
    with pytest.raises(HarnessError, match="not a session id"):
        await lock.acquire(bad.replace("victim", str(tmp_path / "victim")))
    assert len(workspace.execs) == ran


async def test_the_helper_refuses_an_escaping_id_itself(
    lock: SessionLock, workspace: fakes.LockWorkspace, tmp_path: Path
) -> None:
    """The helper does not trust its caller: an absolute id, or a symlink
    planted in the root, never reaches rmtree. Called directly, past the
    check `SessionLock` makes first."""
    from ai.harnesses.experimental._session_lock import HELPER

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("precious")
    root = await lock.root()
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "planted").symlink_to(victim)
    for sid in (str(victim), "../victim", "planted"):
        args = json.dumps({"sid": sid, "token": "t", "stale_after": 0})
        result = await workspace.exec(
            ["python3", "-c", HELPER, "acquire", root, args], timeout=30
        )
        assert result.exit_code != 0
        assert "not a session id" in result.stderr
    assert (victim / "keep.txt").read_text() == "precious"
    assert await lock.holders() == {}
