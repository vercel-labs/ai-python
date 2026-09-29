"""The append-only store, its merge rules, and per-file leases.

Spec §2.9, §4.1.

    data/<projectId>/
    ├── files/<path>.json     one FileRecord per scanned file (primary store)
    ├── runs/<runId>.json     one RunMeta per run
    ├── debug/                raw agent output that failed the contract
    └── reports/              what `export` writes

Every stage is idempotent and additive. Nothing here deletes.
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from deepysec.models import (
    AnalysisEntry,
    CandidateMatch,
    FileRecord,
    FileStatus,
    Finding,
    RunMeta,
    ensure_finding_ids,
    now,
)

if TYPE_CHECKING:
    from pathlib import Path

STALE_LOCK = timedelta(hours=1)


class Store:
    def __init__(self, data_dir: Path, project_id: str) -> None:
        self.project_id = project_id
        self.root = data_dir / project_id
        self.files_dir = self.root / "files"
        self.runs_dir = self.root / "runs"
        self.debug_dir = self.root / "debug"
        self.reports_dir = self.root / "reports"

    # -- records -------------------------------------------------------------

    def record_path(self, file_path: str) -> Path:
        return self.files_dir / f"{file_path}.json"

    def load(self, file_path: str) -> FileRecord | None:
        path = self.record_path(file_path)
        if not path.is_file():
            return None
        record = FileRecord.model_validate_json(path.read_text())
        ensure_finding_ids(record)  # in memory; persisted on the next save
        return record

    def save(self, record: FileRecord) -> None:
        ensure_finding_ids(record)
        _write_atomic(
            self.record_path(record.file_path), record.model_dump_json(indent=2)
        )

    def records(self) -> list[FileRecord]:
        if not self.files_dir.is_dir():
            return []
        out: list[FileRecord] = []
        for path in sorted(self.files_dir.rglob("*.json")):
            record = FileRecord.model_validate_json(path.read_text())
            ensure_finding_ids(record)
            out.append(record)
        return out

    # -- runs ----------------------------------------------------------------

    def save_run(self, run: RunMeta) -> None:
        _write_atomic(
            self.runs_dir / f"{run.run_id}.json", run.model_dump_json(indent=2)
        )

    def load_run(self, run_id: str) -> RunMeta | None:
        path = self.runs_dir / f"{run_id}.json"
        return (
            RunMeta.model_validate_json(path.read_text())
            if path.is_file()
            else None
        )

    def runs(self) -> list[RunMeta]:
        if not self.runs_dir.is_dir():
            return []
        return [
            RunMeta.model_validate_json(p.read_text())
            for p in sorted(self.runs_dir.glob("*.json"))
        ]

    # -- debug ---------------------------------------------------------------

    def write_debug(self, name: str, text: str) -> Path | None:
        """Best effort, never raises: callers are already on an error path."""
        try:
            path = self.debug_dir / name
            _write_atomic(path, text)
            return path
        except OSError:
            return None

    # -- leases (spec §4.1) --------------------------------------------------

    def claim(self, file_path: str, run: RunMeta) -> FileRecord | None:
        """Atomically take a file for this run, or return None.

        Re-read, verify it is still claimable, mark it, write. Two `process`
        runs on one store never both hold a record: the second re-read sees
        the first one's lock.
        """
        record = self.load(file_path)
        if record is None or not self.is_claimable(record):
            return None
        record.status = "processing"
        record.locked_by_run_id = run.run_id
        record.locked_at = now()
        self.save(record)
        return record

    def is_claimable(self, record: FileRecord) -> bool:
        if record.status == "pending":
            return True
        if record.status != "processing":
            return False
        owner = (
            self.load_run(record.locked_by_run_id)
            if record.locked_by_run_id
            else None
        )
        # A same-host run whose process is gone (SIGKILL, OOM) is reclaimed
        # at once rather than after the hour.
        same_host = owner is not None and owner.hostname == socket.gethostname()
        if (
            same_host
            and owner is not None
            and owner.pid
            and not pid_alive(owner.pid)
        ):
            return True
        # A missing lockedAt is "very old", for records that predate the field.
        locked_at = _parse(record.locked_at) if record.locked_at else None
        stale = locked_at is None or datetime.now(UTC) - locked_at > STALE_LOCK
        return stale and (owner is None or owner.phase in ("done", "error"))

    def release(self, record: FileRecord, status: FileStatus) -> None:
        record.status = status
        record.locked_by_run_id = None
        record.locked_at = None
        self.save(record)

    def release_all(self, run_id: str) -> int:
        """On abort: every lock this run still holds goes back to pending."""
        freed = 0
        for record in self.records():
            if (
                record.status == "processing"
                and record.locked_by_run_id == run_id
            ):
                self.release(record, "pending")
                freed += 1
        return freed


# -- merge rules (spec §2.9) ---------------------------------------------------


def merge_candidates(record: FileRecord, found: list[CandidateMatch]) -> int:
    """Re-scan unions; it never deletes a stale candidate."""
    seen = {(c.vuln_slug, c.matched_pattern) for c in record.candidates}
    added = 0
    for candidate in found:
        key = (candidate.vuln_slug, candidate.matched_pattern)
        if key not in seen:
            record.candidates.append(candidate)
            seen.add(key)
            added += 1
    return added


def append_analysis(
    record: FileRecord, entry: AnalysisEntry, findings: list[Finding]
) -> int:
    """Re-process appends history and unions findings by (slug, title).

    A re-run that reports the same issue does not duplicate it, and
    `producedByRunId` stays bound to the run that found it first.
    """
    record.analysis_history.append(entry)
    seen = {f.signature for f in record.findings}
    added = 0
    for finding in findings:
        if finding.signature in seen:
            continue
        finding.produced_by_run_id = entry.run_id
        record.findings.append(finding)
        seen.add(finding.signature)
        added += 1
    ensure_finding_ids(record)
    return added


# -- helpers -------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
