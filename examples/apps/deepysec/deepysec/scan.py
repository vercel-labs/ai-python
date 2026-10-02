"""Stage 1 — scan (spec §3). Free, no AI. Narrows the surface; decides nothing.

Walks the project root the way the SDK's `copy` does — honouring
`.gitignore` and the usual noise directories — runs every active matcher on
each file it applies to, and writes candidates into the FileRecord.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

# Private: the same rules `copy()` uses to decide what travels.
from ai.workspaces.experimental._base import (
    DEFAULT_IGNORE,  # noqa: PLC2701
    walk_uploadable,  # noqa: PLC2701
)
from deepysec.models import FileRecord, RunMeta, now
from deepysec.store import Store, merge_candidates

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepysec.matchers import Matcher

#: deepsec's own generated data is never itself scanned.
SCAN_IGNORE: tuple[str, ...] = (*DEFAULT_IGNORE, ".deepsec")


def scan(
    root: Path,
    store: Store,
    matchers: list[Matcher],
    *,
    log: Callable[[str], None] = print,
) -> RunMeta:
    run = RunMeta(project_id=store.project_id, root_path=str(root), type="scan")
    store.save_run(run)
    for local, relative in walk_uploadable(root, SCAN_IGNORE, gitignore=True):
        applicable = [m for m in matchers if m.applies_to(relative)]
        if not applicable:
            continue
        try:
            content = local.read_text(errors="replace")
        except OSError:
            continue
        found = [c for m in applicable for c in m.match(content, relative)]
        if not found:
            continue
        digest = hashlib.sha256(content.encode()).hexdigest()[:16]
        record = store.load(relative)
        if record is None:
            record = FileRecord(
                file_path=relative,
                project_id=store.project_id,
                last_scanned_at=now(),
                last_scanned_run_id=run.run_id,
                file_hash=digest,
            )
        else:
            # Re-scan MERGES. A file that changed since it was analyzed, or
            # that ended in `error`, is armed for the next `process`; a lock
            # another run holds is left alone.
            if record.status == "error" or (
                record.status == "analyzed" and record.file_hash != digest
            ):
                record.status = "pending"
            record.last_scanned_at = now()
            record.last_scanned_run_id = run.run_id
            record.file_hash = digest
        run.stats.candidates_found += merge_candidates(record, found)
        run.stats.files_scanned += 1
        store.save(record)
    run.phase = "done"
    run.completed_at = now()
    store.save_run(run)
    log(
        f"[scan] {run.stats.files_scanned} files with candidates, "
        + f"{run.stats.candidates_found} new candidates (run {run.run_id})"
    )
    return run
