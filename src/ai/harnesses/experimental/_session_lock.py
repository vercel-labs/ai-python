"""One writer per conversation, enforced in the workspace.

Codex refuses a second writer on a live thread itself. Claude does not, so
two clients can silently write one transcript. This lock gives every
harness codex's behaviour, using nothing but the workspace's own primitives
— a harness acquires it for the conversations it opens, and a returning
client that finds it held gets `SessionBusyError` instead of a corrupted
transcript.

Layout, per harness kind, under the workspace machine's HOME:

    ~/.ai-python/locks/<kind>/<session_id>/holder.json

A lock is a DIRECTORY because `mkdir` is atomic on every POSIX filesystem:
two clients racing for the same conversation cannot both succeed. The
holder file records the pid that owns it, so a lock whose process is gone
is recognised as stale and reclaimed rather than blocking forever. A lock
recorded before its process existed (the pid is filled in after launch) is
trusted only briefly: past `STALE_AFTER` it is a launch that never finished.

Everything here goes through the workspace's own primitives — `home()`,
`write_text`, `read_text`, `exists` — and falls back to a plain argv `exec`
only for what has no primitive: an exclusive `mkdir` (the atomic step),
removal, listing, and a process-liveness probe. Never a shell string.

The harness that acquires a lock keeps it across `detach()` — the process
is still running and still the writer — and releases it on `close()`, once
the process is confirmed gone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import errors

if TYPE_CHECKING:
    from ...workspaces.experimental import _base

STALE_AFTER = 120.0
"""Seconds a pid-less lock is honoured before it counts as abandoned."""

_LIVENESS = "import os, sys\nos.kill(int(sys.argv[1]), 0)\n"


@dataclass(frozen=True)
class Holder:
    """Who holds a conversation open."""

    session_id: str
    pid: int | None
    since: int
    client: str


class SessionLock:
    def __init__(self, workspace: _base.Workspace, kind: str) -> None:
        self._workspace = workspace
        self._kind = kind
        self._root: str | None = None

    async def root(self) -> str:
        if self._root is None:
            home = await self._workspace.home()
            self._root = f"{home}/.ai-python/locks/{self._kind}"
        return self._root

    async def acquire(
        self, session_id: str, *, pid: int | None = None, client: str = ""
    ) -> None:
        """Take the lock, or raise SessionBusyError naming the live holder.

        A stale lock — its pid gone, or pid-less and older than STALE_AFTER
        — is reclaimed. Two clients reclaiming at once still cannot both
        win: the second `mkdir` fails and that client is told busy.
        """
        directory = f"{await self.root()}/{session_id}"
        await self._workspace.exec(
            ["mkdir", "-p", await self.root()], timeout=30
        )
        if await self._mkdir(directory):
            await self._record(directory, pid, client)
            return
        holder = await self._read(session_id, directory)
        if holder is not None and await self._live(holder):
            raise errors.SessionBusyError(session_id, holder.pid)
        await self._workspace.exec(["rm", "-rf", directory], timeout=30)
        if not await self._mkdir(directory):
            raise errors.SessionBusyError(
                session_id, None, "contested while reclaiming a stale lock"
            )
        await self._record(directory, pid, client)

    async def claim_pid(
        self, session_id: str, pid: int | None, client: str = ""
    ) -> None:
        """Fill in the pid once the process exists."""
        if pid is None:
            return
        await self._record(f"{await self.root()}/{session_id}", pid, client)

    async def release(self, session_id: str) -> None:
        await self._workspace.exec(
            ["rm", "-rf", f"{await self.root()}/{session_id}"], timeout=30
        )

    async def holder(self, session_id: str) -> Holder | None:
        """Return the live holder of a conversation, or None when it is free."""
        holder = await self._read(
            session_id, f"{await self.root()}/{session_id}"
        )
        if holder is None or not await self._live(holder):
            return None
        return holder

    async def holders(self) -> dict[str, int | None]:
        """Every conversation with a live holder, and its pid, in one pass."""
        root = await self.root()
        listing = await self._workspace.exec(["ls", root], timeout=30)
        live: dict[str, int | None] = {}
        for session_id in (
            listing.stdout.split() if listing.exit_code == 0 else []
        ):
            holder = await self.holder(session_id)
            if holder is not None:
                live[session_id] = holder.pid
        return live

    # -- plumbing ---------------------------------------------------------

    async def _mkdir(self, directory: str) -> bool:
        return (
            await self._workspace.exec(["mkdir", directory], timeout=30)
        ).exit_code == 0

    async def _record(
        self, directory: str, pid: int | None, client: str
    ) -> None:
        info = json.dumps(
            {"pid": pid, "since": int(time.time()), "client": client}
        )
        await self._workspace.write_text(f"{directory}/holder.json", info)

    async def _read(self, session_id: str, directory: str) -> Holder | None:
        try:
            raw = json.loads(
                await self._workspace.read_text(f"{directory}/holder.json")
            )
        except (FileNotFoundError, ValueError):
            if await self._exists(directory):
                # A directory without a record: a launch that died between
                # mkdir and the write. Nothing can own it.
                return Holder(session_id, None, 0, "")
            return None
        pid = raw.get("pid")
        return Holder(
            session_id,
            int(pid) if isinstance(pid, int) else None,
            int(raw.get("since") or 0),
            str(raw.get("client") or ""),
        )

    async def _exists(self, directory: str) -> bool:
        return await self._workspace.exists(directory)

    async def _live(self, holder: Holder) -> bool:
        if holder.pid is None:
            return (time.time() - holder.since) < STALE_AFTER
        probe = await self._workspace.exec(
            ["python3", "-c", _LIVENESS, str(holder.pid)], timeout=30
        )
        return probe.exit_code == 0
