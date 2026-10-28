"""What `afk pull --files` would write here, decided before anything is written.

Three views of the project: what was pushed (the baseline afk recorded at
push time), what the sandbox has now, and what this directory has now. A
file comes home only if the sandbox changed it. If this directory changed
it too since the push, that is a conflict: your edits would be lost, so it
is listed on its own and needs its own yes.

Without a baseline (a sandbox pushed before afk recorded one) nothing can
be told apart, so every difference is listed as a possible conflict, and
nothing is ever deleted: a file missing there may never have been pushed.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

# Private: the same rules `copy()` uses to decide what travels, so a
# manifest matches exactly what push sent.
from ai.workspaces.experimental._base import (
    DEFAULT_IGNORE,  # noqa: PLC2701
    walk_uploadable,  # noqa: PLC2701
)

if TYPE_CHECKING:
    from pathlib import Path

Action = Literal["add", "update", "delete"]


class Change(BaseModel):
    path: str
    action: Action
    conflict: bool
    """This directory changed the file too since the push (or there is no
    record of what was pushed): writing it would lose local edits."""


def manifest(root: Path) -> dict[str, str]:
    """sha256 of every file that travels, by the same rules push copies with."""
    return {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for path, relative in walk_uploadable(root, DEFAULT_IGNORE)
    }


def plan(
    sandbox: dict[str, str],
    local: dict[str, str],
    baseline: dict[str, str] | None,
) -> list[Change]:
    """The changes to bring home, sorted by path.

    Pure: hashes in, a plan out.
    """
    changes: list[Change] = []
    paths = set(sandbox) | set(baseline if baseline is not None else local)
    for path in sorted(paths):
        there, here = sandbox.get(path), local.get(path)
        if there == here:
            continue  # already the same
        if baseline is not None and there == baseline.get(path):
            continue  # the sandbox did not change it; any difference is yours
        if there is None:
            if here is None or baseline is None:
                continue  # without a record of what was pushed, never delete
            action: Action = "delete"
        else:
            action = "add" if here is None else "update"
        conflict = baseline is None or here != baseline.get(path)
        changes.append(Change(path=path, action=action, conflict=conflict))
    return changes
