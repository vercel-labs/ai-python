"""One writer per conversation, enforced in the workspace.

Codex refuses a second writer on a live thread itself. Claude does not, so
two clients can silently write one transcript. This lock gives every
harness codex's behaviour, using nothing but the workspace — a harness
acquires it for the conversations it opens, and a returning client that
finds it held gets `SessionBusyError` instead of a corrupted transcript.

Layout, per harness kind, under the workspace machine's HOME:

    ~/.ai-python/locks/<kind>/<session_id>/holder.json
    ~/.ai-python/locks/<kind>/.mutex

Every operation is ONE command in the workspace: a small stdlib Python
helper (`HELPER`) that does the whole check-and-set there, so a sandbox
costs one round trip per operation, not one per file. Correctness rests on
three things the helper does:

- **A mutex for the check-and-set.** Acquire, reclaim, claim and release
  run under an exclusive `flock` on `.mutex`, held only for the duration of
  that one command. Two clients judging the same stale lock can no longer
  both delete it and both take it: the second one in finds the first one's
  fresh, live lock and is told busy.
- **A lock never exists without its record.** It is built in a temp dir and
  renamed into place, so a reader never sees a directory with no holder in
  it and mistakes a launch in progress for an abandoned one.
- **A token per holder.** Only the client that took a lock may fill in its
  pid or release it; anyone else's release is a no-op.

A holder is live while its pid exists — a process we may not signal still
exists — and is the process that was recorded (same start time, so a pid
the OS reused for a later process holds nothing). A holder recorded before
its process existed (the pid is filled in after launch) is trusted only
briefly: past `STALE_AFTER` it is a launch that never finished.

The harness that acquires a lock keeps it across `detach()` — the process
is still running and still the writer — and releases it on `close()`, once
the process is confirmed gone.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import errors

if TYPE_CHECKING:
    from ...workspaces.experimental import _base

STALE_AFTER = 120.0
"""Seconds a pid-less lock is honoured before it counts as abandoned."""

HELPER = r"""
import fcntl, json, os, shutil, subprocess, sys, time

op, root, args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
STALE = float(args.get("stale_after", 120))


def started(pid):
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "TZ": "UTC", "LC_ALL": "C"},
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def alive(rec):
    pid = rec.get("pid")
    if not isinstance(pid, int):
        return time.time() - (rec.get("since") or 0) < STALE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # it exists; it is just not ours to signal
    except OSError:
        return False
    want = rec.get("started")
    if want:
        now = started(pid)
        if now is not None and now != want:
            return False  # the pid now names a later process
    return True


def read(sid):
    d = os.path.join(root, sid)
    try:
        with open(os.path.join(d, "holder.json")) as f:
            rec = json.load(f)
        return rec if isinstance(rec, dict) else {"pid": None, "since": 0}
    except FileNotFoundError:
        return {"pid": None, "since": 0} if os.path.isdir(d) else None
    except ValueError:
        return {"pid": None, "since": 0}


def record(pid, client, token, since=None):
    return {
        "pid": pid,
        "started": started(pid) if isinstance(pid, int) else None,
        "since": since or int(time.time()),
        "client": client,
        "token": token,
    }


def write(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)


def live(sid):
    rec = read(sid)
    return rec if rec is not None and alive(rec) else None


os.makedirs(root, exist_ok=True)
out = None
if op == "holders":
    out = {}
    for sid in os.listdir(root):
        if not sid.startswith("."):
            rec = live(sid)
            if rec is not None:
                out[sid] = rec.get("pid")
elif op == "holder":
    out = live(args["sid"])
else:
    with open(os.path.join(root, ".mutex"), "a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        sid, token = args["sid"], args["token"]
        d = os.path.join(root, sid)
        rec = read(sid)
        if op == "acquire":
            if rec is not None and alive(rec):
                out = {"busy": True, "pid": rec.get("pid")}
            else:
                if rec is not None:
                    shutil.rmtree(d, ignore_errors=True)
                tmp = os.path.join(root, "." + sid + "." + token)
                os.makedirs(tmp, exist_ok=True)
                write(
                    os.path.join(tmp, "holder.json"),
                    record(args.get("pid"), args.get("client", ""), token),
                )
                os.rename(tmp, d)
                out = {"busy": False}
        elif op == "claim":
            if rec is not None and rec.get("token") == token:
                write(
                    os.path.join(d, "holder.json"),
                    record(
                        args["pid"], args.get("client", ""), token,
                        since=rec.get("since"),
                    ),
                )
                out = True
            else:
                out = False
        elif op == "release":
            out = rec is not None and rec.get("token") == token
            if out:
                shutil.rmtree(d, ignore_errors=True)
print(json.dumps(out))
"""


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
        # The token of every lock THIS client took: what lets it, and only
        # it, fill in the pid and release.
        self._tokens: dict[str, str] = {}

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
        — is reclaimed.
        """
        token = uuid.uuid4().hex
        out = await self._run(
            "acquire",
            sid=session_id,
            token=token,
            pid=pid,
            client=client,
            stale_after=STALE_AFTER,
        )
        if out.get("busy"):
            raise errors.SessionBusyError(session_id, out.get("pid"))
        self._tokens[session_id] = token

    async def claim_pid(
        self, session_id: str, pid: int | None, client: str = ""
    ) -> None:
        """Fill in the pid once the process exists. Only on our own lock."""
        token = self._tokens.get(session_id)
        if pid is None or token is None:
            return
        await self._run(
            "claim", sid=session_id, token=token, pid=pid, client=client
        )

    async def release(self, session_id: str) -> None:
        """Release a lock this client took; anyone else's is left alone."""
        token = self._tokens.pop(session_id, None)
        if token is None:
            return
        await self._run("release", sid=session_id, token=token)

    async def holder(self, session_id: str) -> Holder | None:
        """Return the live holder of a conversation, or None when it is free."""
        rec = await self._run("holder", sid=session_id, stale_after=STALE_AFTER)
        if not isinstance(rec, dict):
            return None
        pid = rec.get("pid")
        return Holder(
            session_id,
            pid if isinstance(pid, int) else None,
            int(rec.get("since") or 0),
            str(rec.get("client") or ""),
        )

    async def holders(self) -> dict[str, int | None]:
        """Every conversation with a live holder, and its pid, in one pass."""
        out = await self._run("holders", stale_after=STALE_AFTER)
        return {
            str(sid): pid if isinstance(pid, int) else None
            for sid, pid in (out or {}).items()
        }

    async def _run(self, op: str, **args: Any) -> Any:
        result = await self._workspace.exec(
            ["python3", "-c", HELPER, op, await self.root(), json.dumps(args)],
            timeout=30,
        )
        if result.exit_code != 0:
            raise errors.HarnessError(
                f"the session lock could not {op} in the workspace "
                f"(exit {result.exit_code}): "
                + (result.stderr.strip()[-300:] or "no output")
                + ". Harnesses need python3 where the CLI runs."
            )
        return json.loads(result.stdout.strip().splitlines()[-1])
