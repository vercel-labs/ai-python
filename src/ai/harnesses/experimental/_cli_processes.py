"""The processes running in a workspace, as `ps` reports them.

How a harness finds the CLIs it did not launch: a `claude` or `codex` typed
in a terminal holds no SDK lock. Through the workspace, so it reads the
machine the CLI runs on, this one or a sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...workspaces.experimental import _base


@dataclass(frozen=True)
class Process:
    pid: int
    started: str
    """When it started, as `ps` prints it, in UTC: with the pid, which
    process this is, not a later one that reuses the pid."""
    argv: list[str]
    """Split on spaces: `ps` joins the argv, so an argument with a space in it
    comes back as several."""


async def processes(workspace: _base.Workspace) -> dict[int, Process]:
    """Every process in the workspace, by pid; empty when `ps` cannot run."""
    result = await workspace.exec(
        ["ps", "-eo", "pid=,lstart=,args="],
        env={"TZ": "UTC", "LC_ALL": "C"},
        timeout=30,
    )
    if result.exit_code != 0:
        return {}
    found: dict[int, Process] = {}
    for line in result.stdout.splitlines():
        pid, _, rest = line.strip().partition(" ")
        if not pid.isdigit():
            continue
        rest = rest.lstrip()
        # lstart is fixed width: "Fri Oct  2 16:20:16 2026".
        found[int(pid)] = Process(int(pid), rest[:24], rest[24:].split())
    return found
