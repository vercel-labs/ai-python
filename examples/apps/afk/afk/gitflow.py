"""Push and pull through git: a branch in the sandbox, a branch at home.

In a git repository, `push` records your working tree as it is —
uncommitted and untracked (not ignored) files included — as a commit on top
of HEAD, without touching your branch, your index or your files: the BASE.
The sandbox gets a repository whose branch `afk/<label>` starts from that
tree. The agent works there and may commit as it likes.

`pull --files` commits whatever it left uncommitted, and brings its commits
home as YOUR branch `afk/<label>`, built on the base. Your checkout is not
touched: you look at it and merge it like any branch, and a conflict with
what you did meanwhile is git's, resolved with git.

Without `--git-history` the sandbox repository has one commit, the tree as
pushed: the agent sees no history (smaller, faster). With it, the whole
history travels, as a bundle, for tasks that need `git log` or `blame`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ai.workspaces.experimental import copy

if TYPE_CHECKING:
    from ai.workspaces.experimental import Workspace

IDENTITY = ("-c", "user.name=afk", "-c", "user.email=afk@localhost")
"""Who commits in the sandbox, unless the agent says otherwise."""
PATCH = "/tmp/afk-pull.patch"


class GitPush(BaseModel):
    """What a push recorded, to bring the work home as a branch."""

    branch: str
    base: str
    """The commit at home the work starts from (your tree as pushed)."""
    root: str
    """The commit in the sandbox the work starts from: `base` itself with
    `--git-history`, a one-commit copy of its tree without."""
    history: bool = False


class GitError(Exception):
    """A git step failed; the message is git's own."""


def branch_for(label: str) -> str:
    """`afk/<label>`, with what git refuses in a branch name replaced."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-.") or "push"
    return f"afk/{name}"


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        check=False,
    )
    if done.returncode != 0:
        raise GitError(
            f"git {' '.join(args[:2])} failed: "
            + (done.stderr.strip() or done.stdout.strip())[-400:]
        )
    return done.stdout.strip()


def repo_root(cwd: Path) -> Path | None:
    """The repository `cwd` is the top of, or None: the git flow is for a
    push of a whole repository; any other directory pulls files instead."""
    try:
        top = _git(cwd, "rev-parse", "--show-toplevel")
    except (GitError, OSError):
        return None
    return cwd if Path(top).resolve() == cwd.resolve() else None


def _identity(cwd: Path) -> dict[str, str]:
    """Your git identity if you have one, else afk's: commit-tree needs one."""
    try:
        _git(cwd, "config", "user.email")
        return {}
    except GitError:
        return {
            "GIT_AUTHOR_NAME": "afk",
            "GIT_AUTHOR_EMAIL": "afk@localhost",
            "GIT_COMMITTER_NAME": "afk",
            "GIT_COMMITTER_EMAIL": "afk@localhost",
        }


def record_base(cwd: Path, branch: str) -> str:
    """Commit the working tree as it is, on top of HEAD, out of the way.

    A temporary index, so yours is untouched; `commit-tree`, so no branch
    moves; kept under refs/afk/, so git does not collect it.
    """
    with tempfile.TemporaryDirectory(prefix="afk-index-") as tmp:
        env = {**_identity(cwd), "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        try:
            head: str | None = _git(cwd, "rev-parse", "--verify", "-q", "HEAD")
        except GitError:
            head = None  # a repository with no commit yet
        if head:
            _git(cwd, "read-tree", head, env=env)
        _git(cwd, "add", "-A", env=env)
        tree = _git(cwd, "write-tree", env=env)
        parents = ["-p", head] if head else []
        base = _git(
            cwd,
            "commit-tree",
            tree,
            *parents,
            "-m",
            f"afk: your tree, as pushed to {branch}",
            env=env,
        )
    _git(cwd, "update-ref", f"refs/{branch}/base", base)
    return base


async def _run(ws: Workspace, *args: str) -> str:
    done = await ws.exec(["git", *IDENTITY, *args], timeout=600)
    if done.exit_code != 0:
        raise GitError(
            f"git {' '.join(args[:2])} failed in the sandbox: "
            + (done.stderr.strip() or done.stdout.strip())[-400:]
        )
    return done.stdout.strip()


async def start(
    ws: Workspace, cwd: Path, *, branch: str, history: bool
) -> GitPush:
    """Record the base at home and start the branch in the sandbox, whose
    tree was already copied there."""
    base = await asyncio.to_thread(record_base, cwd, branch)
    if not history:
        await _run(ws, "init", "-q", "-b", branch)
        await _run(ws, "add", "-A")
        await _run(ws, "commit", "-q", "--allow-empty", "-m", "afk: as pushed")
        root = await _run(ws, "rev-parse", "HEAD")
        return GitPush(branch=branch, base=base, root=root, history=False)
    with tempfile.TemporaryDirectory(prefix="afk-bundle-") as tmp:
        bundle = Path(tmp) / "base.bundle"
        await asyncio.to_thread(
            _git, cwd, "bundle", "create", str(bundle), f"refs/{branch}/base"
        )
        await copy(tmp, ws / ".afk-in")
    # Born under another name: git will not fetch into the branch HEAD
    # names while it has no commit.
    await _run(ws, "init", "-q", "-b", "afk-unborn")
    await _run(
        ws, "fetch", "-q", ".afk-in/base.bundle", f"{base}:refs/heads/{branch}"
    )
    await _run(ws, "symbolic-ref", "HEAD", f"refs/heads/{branch}")
    await ws.exec(["rm", "-rf", ".afk-in"], timeout=60)
    # The index from the base; files a push does not copy (build output a
    # repository tracks) come back from it, so they do not read as deleted.
    await _run(ws, "reset", "-q", "--mixed", branch)
    await _run(ws, "checkout", "-q", "--", ".")
    return GitPush(branch=branch, base=base, root=base, history=True)


async def bring_home(
    ws: Workspace, cwd: Path, push: GitPush, *, label: str
) -> int:
    """Commit what the agent left uncommitted, and make its commits your
    branch `push.branch`, on the base. Returns how many commits it has.
    Your checkout is never touched."""
    await _run(ws, "add", "-A")
    staged = await ws.exec(["git", "diff", "--cached", "--quiet"], timeout=60)
    if staged.exit_code != 0:
        await _run(
            ws, "commit", "-q", "-m", f"afk: left uncommitted in {label}"
        )
    count = int(await _run(ws, "rev-list", "--count", f"{push.root}..HEAD"))
    if count == 0:
        return 0
    linear = await ws.exec(
        ["git", "merge-base", "--is-ancestor", push.root, "HEAD"], timeout=60
    )
    if linear.exit_code == 0:
        made = await ws.exec(
            [
                "sh",
                "-c",
                'git format-patch --binary --stdout "$1"..HEAD > "$2"',
                "sh",
                push.root,
                PATCH,
            ],
            timeout=600,
        )
    else:
        # History rewritten past the start: the net change, as one commit.
        made = await ws.exec(
            [
                "sh",
                "-c",
                'git diff --binary "$1" HEAD > "$2"',
                "sh",
                push.root,
                PATCH,
            ],
            timeout=600,
        )
        count = 1
    if made.exit_code != 0:
        raise GitError(f"could not export the work: {made.stderr.strip()}")
    patch = await ws.read_bytes(PATCH)
    await asyncio.to_thread(
        _apply, cwd, push, patch, squashed=linear.exit_code != 0, label=label
    )
    return count


def _apply(
    cwd: Path, push: GitPush, patch: bytes, *, squashed: bool, label: str
) -> None:
    """Replay the work onto the base in a worktree of its own."""
    checked_out = _git(cwd, "worktree", "list", "--porcelain")
    if f"branch refs/heads/{push.branch}\n" in checked_out + "\n":
        raise GitError(
            f"{push.branch} is checked out; switch away from it to pull "
            "into it"
        )
    tmp = Path(tempfile.mkdtemp(prefix="afk-pull-"))
    shutil.rmtree(tmp)
    env = _identity(cwd)
    _git(cwd, "worktree", "add", "-q", "--detach", str(tmp), push.base)
    try:
        patch_file = tmp.parent / f"{tmp.name}.patch"
        patch_file.write_bytes(patch)
        try:
            if squashed:
                _git(tmp, "apply", "--index", "--binary", str(patch_file))
                _git(
                    tmp,
                    "commit",
                    "-q",
                    "-m",
                    f"afk: the work in {label}",
                    env=env,
                )
            else:
                _git(tmp, "am", "-q", "--keep-cr", str(patch_file), env=env)
        finally:
            patch_file.unlink(missing_ok=True)
        _git(cwd, "branch", "-f", push.branch, _git(tmp, "rev-parse", "HEAD"))
    finally:
        with contextlib.suppress(GitError):
            _git(cwd, "worktree", "remove", "--force", str(tmp))
