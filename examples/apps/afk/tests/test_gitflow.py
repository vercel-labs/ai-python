"""The git round trip, with real git: a `Local` workspace stands in for the
sandbox — the same `exec`, `copy` and `read_bytes` a VercelSandbox has."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from afk import gitflow

from ai.workspaces.experimental import Local, copy

if TYPE_CHECKING:
    from pathlib import Path


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    git(home, "init", "-q", "-b", "main")
    (home / "app.py").write_text("print('v1')\n")
    (home / ".gitignore").write_text("*.log\n")
    git(home, "add", "-A")
    git(home, "commit", "-q", "-m", "first")
    (home / "app.py").write_text("print('v2')\n")
    git(home, "commit", "-qam", "second")
    # Work you have not committed travels too; your index stays as it is.
    (home / "app.py").write_text("print('v3, uncommitted')\n")
    (home / "notes.md").write_text("untracked\n")
    (home / "debug.log").write_text("ignored\n")
    return home


async def _push(
    repo: Path, tmp_path: Path, *, history: bool
) -> tuple[Local, gitflow.GitPush]:
    sandbox = Local(tmp_path / "sandbox")
    await sandbox.open()
    await copy(repo, sandbox / ".")
    push = await gitflow.start(sandbox, repo, branch="afk/t1", history=history)
    return sandbox, push


def _agent_works(sandbox: Path) -> None:
    (sandbox / "app.py").write_text("print('v4, by the agent')\n")
    git(sandbox, "commit", "-qam", "agent: v4")
    (sandbox / "new.py").write_text("print('new')\n")  # left uncommitted


@pytest.mark.parametrize("history", [False, True])
async def test_the_work_comes_home_as_a_branch_and_your_checkout_is_untouched(
    repo: Path,
    tmp_path: Path,
    history: bool,  # noqa: FBT001 (a parametrized fixture)
) -> None:
    status_before = git(repo, "status", "--porcelain")
    sandbox, push = await _push(repo, tmp_path, history=history)
    there = tmp_path / "sandbox"
    assert git(there, "rev-parse", "--abbrev-ref", "HEAD") == "afk/t1"
    seen = git(there, "log", "--format=%s")
    if history:
        assert "first" in seen and "second" in seen, seen
    else:
        assert seen == "afk: as pushed"
    assert (there / "notes.md").exists(), "untracked work travels"
    assert git(there, "status", "--porcelain") == "", "nothing reads changed"

    _agent_works(there)
    count = await gitflow.bring_home(sandbox, repo, push, label="t1")
    assert count == 2
    assert git(repo, "status", "--porcelain") == status_before
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(repo, "log", "--format=%s", "-3", "afk/t1").splitlines() == [
        "afk: left uncommitted in t1",
        "agent: v4",
        "afk: your tree, as pushed to afk/t1",
    ]
    assert git(repo, "show", "afk/t1:app.py") == "print('v4, by the agent')"
    assert git(repo, "show", "afk/t1:notes.md") == "untracked"
    assert git(repo, "rev-parse", "afk/t1~2") == push.base
    assert git(repo, "rev-parse", f"{push.base}^") == git(
        repo, "rev-parse", "main"
    )
    await sandbox.close()


async def test_a_change_on_both_sides_is_a_git_conflict(
    repo: Path, tmp_path: Path
) -> None:
    sandbox, push = await _push(repo, tmp_path, history=False)
    _agent_works(tmp_path / "sandbox")
    await gitflow.bring_home(sandbox, repo, push, label="t1")
    # Meanwhile, at home (your work committed, as before any merge):
    (repo / "app.py").write_text("print('v4, by you')\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "yours")
    merged = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "afk/t1"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert merged.returncode != 0 and "CONFLICT" in merged.stdout
    assert "<<<<<<<" in (repo / "app.py").read_text()
    await sandbox.close()


async def test_no_work_is_no_branch(repo: Path, tmp_path: Path) -> None:
    sandbox, push = await _push(repo, tmp_path, history=False)
    assert await gitflow.bring_home(sandbox, repo, push, label="t1") == 0
    assert "afk/t1" not in git(repo, "branch", "--list", "afk/t1")
    await sandbox.close()


async def test_a_branch_you_have_checked_out_is_not_moved(
    repo: Path, tmp_path: Path
) -> None:
    sandbox, push = await _push(repo, tmp_path, history=False)
    _agent_works(tmp_path / "sandbox")
    await gitflow.bring_home(sandbox, repo, push, label="t1")
    git(repo, "stash", "-u")
    git(repo, "checkout", "-q", "afk/t1")
    (tmp_path / "sandbox" / "more.py").write_text("x = 1\n")
    with pytest.raises(gitflow.GitError, match="checked out"):
        await gitflow.bring_home(sandbox, repo, push, label="t1")
    await sandbox.close()


def test_a_label_becomes_a_branch_name_git_accepts() -> None:
    assert gitflow.branch_for("fix the tests!") == "afk/fix-the-tests"
    assert gitflow.branch_for("..") == "afk/push"


def test_only_the_top_of_a_repository_uses_git(
    repo: Path, tmp_path: Path
) -> None:
    (repo / "sub").mkdir()
    assert gitflow.repo_root(repo) == repo
    assert gitflow.repo_root(repo / "sub") is None
    assert gitflow.repo_root(tmp_path) is None
