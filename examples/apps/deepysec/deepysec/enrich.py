"""Stage 4 — enrich (spec §6).

Host-only by design: git history lives here, not in a sandbox `copy` shipped
without `.git`. Turns a list of vulnerabilities into a list with names attached.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from deepysec.models import Committer, GitInfo, now

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepysec.store import Store


def recent_committers(
    root: Path, file_path: str, limit: int = 5
) -> list[Committer]:
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                f"-n{limit}",
                "--format=%an%x1f%ae%x1f%aI",
                "--",
                file_path,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if out.returncode != 0:
        return []
    committers = []
    for line in out.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3:
            committers.append(
                Committer(name=parts[0], email=parts[1], date=parts[2])
            )
    return committers


def enrich(
    root: Path,
    store: Store,
    *,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> int:
    """Set `gitInfo` on records that have findings, keeping what is there."""
    count = 0
    for record in store.records():
        if not record.findings or (record.git_info is not None and not force):
            continue
        record.git_info = GitInfo(
            recent_committers=recent_committers(root, record.file_path),
            enriched_at=now(),
        )
        store.save(record)
        count += 1
    log(f"[enrich] {count} files enriched with git history")
    return count
