"""The invariants a rewrite must keep (spec §2.4, §2.9, §4.1, §5, §13)."""

from __future__ import annotations

import os
import socket
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
from deepysec.models import (
    AnalysisEntry,
    CandidateMatch,
    FileRecord,
    Finding,
    Revalidation,
    RunMeta,
    Verdict,
    ensure_finding_ids,
    finding_id,
    now,
    split_evenly,
)
from deepysec.revalidate import reconcile
from deepysec.store import Store, append_analysis, merge_candidates
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path


def loaded(store: Store, path: str) -> FileRecord:
    loaded = store.load(path)
    assert loaded is not None
    return loaded


def record(path: str = "src/a.ts", **kw: Any) -> FileRecord:
    return FileRecord(
        file_path=path,
        project_id="p",
        last_scanned_at=now(),
        last_scanned_run_id="scan",
        file_hash="h",
        **kw,
    )


def finding(title: str, slug: str = "xss", **kw: Any) -> Finding:
    return Finding(
        severity="HIGH", vuln_slug=slug, title=title, description="d", **kw
    )


# -- identity ------------------------------------------------------------------


def test_finding_id_is_deterministic_and_path_normalized() -> None:
    a = finding_id("p", "src/a.ts", "SQLi in getUser")
    assert a.startswith("finding_") and len(a) == len("finding_") + 16
    assert a == finding_id("p", "./src/a.ts", "SQLi in getUser")
    assert a == finding_id("p", "src\\a.ts", "SQLi in getUser")
    assert a != finding_id("q", "src/a.ts", "SQLi in getUser")
    assert a != finding_id("p", "src/a.ts", "SQLi in getUser ")


def test_backfill_salts_same_title_collisions_stably() -> None:
    r = record(findings=[finding("dup", "xss"), finding("dup", "rce")])
    assert ensure_finding_ids(r)
    first, second = (f.finding_id for f in r.findings)
    assert first != second
    r2 = record(findings=[finding("dup", "xss"), finding("dup", "rce")])
    ensure_finding_ids(r2)
    assert [f.finding_id for f in r2.findings] == [first, second]
    assert not ensure_finding_ids(r2)


# -- merge rules ---------------------------------------------------------------


def test_reprocess_appends_history_and_unions_findings() -> None:
    r = record()
    entry = AnalysisEntry(
        run_id="r1",
        investigated_at=now(),
        duration_ms=1,
        agent_type="claude-code",
        model="m",
        finding_count=1,
    )
    assert append_analysis(r, entry, [finding("A")]) == 1
    entry2 = entry.model_copy(update={"run_id": "r2", "agent_type": "codex"})
    assert append_analysis(r, entry2, [finding("A"), finding("B")]) == 1
    assert [e.run_id for e in r.analysis_history] == ["r1", "r2"]
    assert [f.title for f in r.findings] == ["A", "B"]
    assert (
        r.findings[0].produced_by_run_id == "r1"
    )  # first discovery, never updated
    assert r.findings[1].produced_by_run_id == "r2"
    assert all(f.finding_id for f in r.findings)


def test_rescan_unions_candidates() -> None:
    r = record()
    c = CandidateMatch(
        vuln_slug="xss", line_numbers=[1], snippet="s", matched_pattern="p"
    )
    assert merge_candidates(r, [c]) == 1
    assert (
        merge_candidates(r, [c, c.model_copy(update={"matched_pattern": "q"})])
        == 1
    )
    assert len(r.candidates) == 2


def test_on_disk_shape_is_deepsecs(tmp_path: Path) -> None:
    store = Store(tmp_path, "p")
    r = record(findings=[finding("A")])
    store.save(r)
    text = store.record_path("src/a.ts").read_text()
    assert (
        '"filePath"' in text and '"vulnSlug"' in text and '"findingId"' in text
    )
    assert "file_path" not in text
    assert store.load("src/a.ts") == store.records()[0]


# -- leases --------------------------------------------------------------------


def run_meta(**kw: Any) -> RunMeta:
    return RunMeta(project_id="p", root_path="/x", type="process", **kw)


def test_claim_is_exclusive(tmp_path: Path) -> None:
    store = Store(tmp_path, "p")
    store.save(record())
    me = run_meta(pid=os.getpid(), hostname=socket.gethostname())
    store.save_run(me)
    other = run_meta(pid=os.getpid(), hostname=socket.gethostname())
    store.save_run(other)
    assert store.claim("src/a.ts", me) is not None
    assert store.claim("src/a.ts", other) is None  # held by a live run
    assert store.claim("src/a.ts", me) is None  # not even by the holder twice


def test_stale_lock_of_finished_run_is_reclaimed(tmp_path: Path) -> None:
    store = Store(tmp_path, "p")
    dead = run_meta(phase="done", hostname="elsewhere", pid=1)
    store.save_run(dead)
    old = (
        (datetime.now(UTC) - timedelta(hours=2))
        .isoformat()
        .replace("+00:00", "Z")
    )
    store.save(
        record(status="processing", locked_by_run_id=dead.run_id, locked_at=old)
    )
    assert store.is_claimable(loaded(store, "src/a.ts"))
    fresh = now()
    store.save(
        record(
            status="processing", locked_by_run_id=dead.run_id, locked_at=fresh
        )
    )
    assert not store.is_claimable(loaded(store, "src/a.ts"))


def test_stale_lock_of_running_run_elsewhere_waits(tmp_path: Path) -> None:
    store = Store(tmp_path, "p")
    running = run_meta(phase="running", hostname="elsewhere", pid=1)
    store.save_run(running)
    old = (
        (datetime.now(UTC) - timedelta(hours=2))
        .isoformat()
        .replace("+00:00", "Z")
    )
    store.save(
        record(
            status="processing", locked_by_run_id=running.run_id, locked_at=old
        )
    )
    assert not store.is_claimable(loaded(store, "src/a.ts"))


def test_same_host_dead_pid_is_reclaimed_immediately(tmp_path: Path) -> None:
    store = Store(tmp_path, "p")
    crashed = run_meta(
        phase="running", hostname=socket.gethostname(), pid=2**22 - 7
    )
    store.save_run(crashed)
    store.save(
        record(
            status="processing",
            locked_by_run_id=crashed.run_id,
            locked_at=now(),
        )
    )
    assert store.is_claimable(loaded(store, "src/a.ts"))


def test_missing_run_and_missing_locked_at_count_as_very_old(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path, "p")
    store.save(
        record(status="processing", locked_by_run_id="gone", locked_at=None)
    )
    assert store.is_claimable(loaded(store, "src/a.ts"))


def test_release_all_frees_only_this_runs_locks(tmp_path: Path) -> None:
    store = Store(tmp_path, "p")
    me = run_meta(pid=os.getpid(), hostname=socket.gethostname())
    other = run_meta(pid=os.getpid(), hostname=socket.gethostname())
    store.save_run(me)
    store.save_run(other)
    store.save(record("a.ts"))
    store.save(record("b.ts"))
    assert store.claim("a.ts", me) and store.claim("b.ts", other)
    assert store.release_all(me.run_id) == 1
    assert loaded(store, "a.ts").status == "pending"
    assert loaded(store, "b.ts").status == "processing"


# -- cost attribution ----------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "parts"), [(10, 3), (0, 4), (7, 7), (1, 2), (100, 1)]
)
def test_split_evenly_sums_exactly(total: int, parts: int) -> None:
    shares = split_evenly(total, parts)
    assert len(shares) == parts and sum(shares) == total
    assert max(shares) - min(shares) <= 1


# -- revalidation contract -----------------------------------------------------


def test_agent_can_never_emit_accepted_risk() -> None:
    with pytest.raises(ValidationError):
        Verdict.model_validate(
            {
                "finding_id": "finding_x",
                "verdict": "accepted-risk",
                "reasoning": "no",
            }
        )
    Revalidation(
        verdict="accepted-risk",
        reasoning="manual",
        revalidated_at=now(),
        run_id="r",
        model="human",
    )


def test_duplicate_invariant() -> None:
    r = record(findings=[finding("A"), finding("B"), finding("C")])
    ensure_finding_ids(r)
    a, b, c = (f.finding_id or "" for f in r.findings)
    items = [(r, f) for f in r.findings]
    verdicts = [
        Verdict(finding_id=a, verdict="true-positive", reasoning="real"),
        Verdict(
            finding_id=b, verdict="duplicate", reasoning="same", duplicate_of=a
        ),
        Verdict(
            finding_id=c, verdict="duplicate", reasoning="same", duplicate_of=b
        ),  # points at a duplicate
    ]
    applied, rejected = reconcile(items, verdicts, run_id="r", model="m")
    assert applied == 2 and rejected == [c]
    assert r.findings[2].revalidation is None
    self_ref = [
        Verdict(
            finding_id=c, verdict="duplicate", reasoning="me", duplicate_of=c
        )
    ]
    assert reconcile(items, self_ref, run_id="r", model="m") == (0, [c])
    missing = [
        Verdict(
            finding_id=c,
            verdict="duplicate",
            reasoning="?",
            duplicate_of="finding_nope",
        )
    ]
    assert reconcile(items, missing, run_id="r", model="m") == (0, [c])


def test_manual_accepted_risk_is_never_overwritten() -> None:
    f = finding("A")
    f.revalidation = Revalidation(
        verdict="accepted-risk",
        reasoning="known",
        revalidated_at=now(),
        run_id="h",
        model="human",
    )
    r = record(findings=[f])
    ensure_finding_ids(r)
    applied, _ = reconcile(
        [(r, f)],
        [
            Verdict(
                finding_id=f.finding_id or "",
                verdict="true-positive",
                reasoning="x",
            )
        ],
        run_id="r",
        model="m",
    )
    assert applied == 0 and f.revalidation.verdict == "accepted-risk"


def test_verdicts_for_unknown_ids_are_ignored() -> None:
    r = record(findings=[finding("A")])
    ensure_finding_ids(r)
    applied, rejected = reconcile(
        [(r, r.findings[0])],
        [
            Verdict(
                finding_id="finding_stranger", verdict="fixed", reasoning="?"
            )
        ],
        run_id="r",
        model="m",
    )
    assert (applied, rejected) == (0, []) and r.findings[0].revalidation is None
