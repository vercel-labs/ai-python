"""A sandbox's lifecycle against a fake platform: what close, stop and a
reconnect do, and what `info()` reports, without a VM."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai.workspaces.experimental import VercelSandbox
from ai.workspaces.experimental.errors import WorkspaceGoneError


class FakeSandbox:
    def __init__(self, name: str, *, status: str = "running") -> None:
        self.name = name
        self.status = status
        self.current_snapshot_id: str | None = None
        self.project_id = "prj_x"
        self.region = "iad1"
        self.vcpus = 2
        self.memory = 4096
        self.created_at = 1
        self.calls: list[str] = []
        self.fs = SimpleNamespace(mkdir=self._mkdir)
        self.execution_time_limit: float | None = None
        self.sessions: list[dict[str, Any]] = [
            {
                "id": "s1",
                "status": "stopped",
                "region": "iad1",
                "vcpus": 2,
                "memory": 4096,
                "startedAt": 1000,
                "stoppedAt": 61000,
                "duration": 60000,
                "activeCpuDurationMs": 6000,
                "networkTransfer": {"ingress": 10, "egress": 2048},
            },
            {"id": "s2", "status": "running", "startedAt": 70000},
        ]

        async def request_json(*args: Any, **kwargs: Any) -> Any:
            return {"sessions": self.sessions, "pagination": {"next": None}}

        async def credentials() -> Any:
            return SimpleNamespace(project_id="prj_x")

        self._service = SimpleNamespace(
            api_client=SimpleNamespace(
                _request_json=request_json, _credentials_factory=credentials
            )
        )

    async def _mkdir(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def stop(self) -> None:
        self.calls.append("stop")
        self.status = "stopped"
        self.current_snapshot_id = "snap_1"

    async def destroy(self, *, delete_orphan_snapshots: bool = False) -> None:
        self.calls.append(f"destroy(orphans={delete_orphan_snapshots})")
        self.status = "destroyed"

    async def session(self) -> None:
        self.calls.append("resume")
        self.status = "running"


class FakeApi:
    def __init__(self) -> None:
        self.sandboxes: dict[str, FakeSandbox] = {}

    async def create_sandbox(self, **options: Any) -> FakeSandbox:
        box = FakeSandbox(f"sb{len(self.sandboxes) + 1}")
        self.sandboxes[box.name] = box
        return box

    async def get_sandbox(self, *, name: str) -> FakeSandbox:
        box = self.sandboxes.get(name)
        if box is None or box.status == "destroyed":
            raise NotFoundError(404)
        return box

    async def get_snapshot(self, *, snapshot_id: str) -> Any:
        return SimpleNamespace(
            size_bytes=800_000_000, created_at=5, expires_at=2_592_000_005
        )


class NotFoundError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    fake = FakeApi()
    monkeypatch.setattr(VercelSandbox, "_api", lambda self: fake)

    async def no_egress_record(self: Any) -> set[str]:
        return set()

    monkeypatch.setattr(VercelSandbox, "_read_egress_record", no_egress_record)
    return fake


async def test_closing_a_sandbox_you_do_not_keep_destroys_it(
    api: FakeApi,
) -> None:
    """Stopping was not ending: a persistent sandbox stays, and its snapshot
    is billed for 30 days. Measured: 1,324 left behind, 1,309 GB."""
    async with VercelSandbox() as ws:
        name = ws.name
    box = api.sandboxes[str(name)]
    assert box.calls == ["destroy(orphans=True)"]


async def test_a_kept_sandbox_is_left_running(api: FakeApi) -> None:
    async with VercelSandbox(keep=True) as ws:
        name = ws.name
    assert api.sandboxes[str(name)].calls == []


async def test_stop_keeps_the_files_and_a_reconnect_resumes(
    api: FakeApi,
) -> None:
    ws = VercelSandbox(keep=True)
    await ws.open()
    name = str(ws.name)
    await ws.stop()
    box = api.sandboxes[name]
    assert box.calls == ["stop"] and box.status == "stopped"
    async with VercelSandbox(name=name) as again:
        assert again.name == name
    assert box.calls == ["stop", "resume"]
    assert box.status == "running"


async def test_a_stopped_sandbox_without_a_snapshot_is_gone(
    api: FakeApi,
) -> None:
    api.sandboxes["old"] = FakeSandbox("old", status="stopped")
    with pytest.raises(WorkspaceGoneError, match="no snapshot"):
        async with VercelSandbox(name="old"):
            pass


async def test_a_failed_sandbox_is_gone(api: FakeApi) -> None:
    api.sandboxes["bad"] = FakeSandbox("bad", status="failed")
    with pytest.raises(WorkspaceGoneError, match="cannot be resumed"):
        async with VercelSandbox(name="bad"):
            pass


async def test_info_describes_a_stopped_sandbox_without_resuming_it(
    api: FakeApi,
) -> None:
    box = FakeSandbox("parked", status="stopped")
    box.current_snapshot_id = "snap_9"
    api.sandboxes["parked"] = box
    info = await VercelSandbox(name="parked").info()
    assert box.calls == [], "asking must not resume it"
    assert info.status == "stopped" and info.resumable
    assert info.snapshot is not None and info.snapshot.size_bytes == 800_000_000
    used, running = info.sessions
    assert (used.duration_ms, used.active_cpu_ms, used.egress_bytes) == (
        60000,
        6000,
        2048,
    )
    assert running.status == "running" and running.duration_ms is None


async def test_info_on_a_sandbox_that_no_longer_exists_is_gone(
    api: FakeApi,
) -> None:
    with pytest.raises(WorkspaceGoneError):
        await VercelSandbox(name="nowhere").info()


async def test_destroy_deletes_a_stopped_sandbox_without_resuming_it(
    api: FakeApi,
) -> None:
    box = FakeSandbox("parked", status="stopped")
    box.current_snapshot_id = "snap_9"
    api.sandboxes["parked"] = box
    await VercelSandbox(name="parked").destroy()
    assert box.calls == ["destroy(orphans=True)"]


async def test_a_sandbox_that_ran_out_of_time_reads_as_expired(
    api: FakeApi,
) -> None:
    box = FakeSandbox("timed", status="stopped")
    box.current_snapshot_id = "snap_1"
    box.execution_time_limit = 60
    box.sessions = box.sessions[:1]  # one session, which ran its 60s
    api.sandboxes["timed"] = box
    info = await VercelSandbox(name="timed").info()
    assert info.expired
