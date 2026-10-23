"""What a project's `.gitignore` excludes never leaves your machine.

These need no workspace: `walk_uploadable` decides what `copy()` sends, and
the decision is visible before anything moves. The last tests hold the whole
reading up against git itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from ai.workspaces.experimental._base import walk_uploadable


def put(root: Path, *paths: str) -> None:
    for path in paths:
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"{path}\n")


def files(
    root: Path, *, ignore: Sequence[str] = (), gitignore: bool = True
) -> set[str]:
    return {
        relative
        for _, relative in walk_uploadable(root, ignore, gitignore=gitignore)
    }


def test_ignored_files_stay_behind(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n.env\n")
    put(tmp_path, "debug.log", ".env", "keep.py", "deep/trace.log")

    assert files(tmp_path) == {".gitignore", "keep.py"}


def test_the_flag_ships_everything(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n.env\n")
    put(tmp_path, "debug.log", ".env", "keep.py")

    assert files(tmp_path, gitignore=False) == {
        ".gitignore",
        "debug.log",
        ".env",
        "keep.py",
    }


def test_ignore_globs_and_gitignore_both_apply(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n")
    put(tmp_path, "debug.log", "notes.md", "keep.py")

    assert files(tmp_path, ignore=["*.md"]) == {".gitignore", "keep.py"}


def test_a_trailing_slash_means_directories_only(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build/\n")
    put(tmp_path, "build/out.js", "src/build", "src/build.py")

    assert files(tmp_path) == {".gitignore", "src/build", "src/build.py"}


def test_a_bare_name_matches_at_any_depth(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secrets\n")
    put(
        tmp_path,
        "secrets/key",
        "deep/secrets/key",
        "flat/secrets",
        "deep/ok.txt",
    )

    assert files(tmp_path) == {".gitignore", "deep/ok.txt"}


def test_a_leading_slash_anchors_to_the_file(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("/todo.md\n")
    put(tmp_path, "todo.md", "docs/todo.md")

    assert files(tmp_path) == {".gitignore", "docs/todo.md"}


def test_a_slash_in_the_middle_anchors_too(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("docs/drafts\n")
    put(tmp_path, "docs/drafts/a.md", "other/docs/drafts/a.md")

    assert files(tmp_path) == {".gitignore", "other/docs/drafts/a.md"}


def test_wildcards_do_not_cross_a_slash(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("a*c\n")
    put(tmp_path, "abc", "a/b/c")

    assert files(tmp_path) == {".gitignore", "a/b/c"}


def test_negation_reinstates(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n!important.log\n")
    put(tmp_path, "important.log", "other.log")

    assert files(tmp_path) == {".gitignore", "important.log"}


def test_negation_cannot_reach_into_an_excluded_directory(
    tmp_path: Path,
) -> None:
    """gitignore(5): a parent that is excluded takes its children with it."""
    (tmp_path / ".gitignore").write_text("logs/\n!logs/keep.log\n")
    put(tmp_path, "logs/keep.log", "logs/other.log")

    assert files(tmp_path) == {".gitignore"}


def test_the_everything_but_idiom(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*\n!*/\n!*.py\n!.gitignore\n")
    put(tmp_path, "a.py", "c.txt", "sub/b.py", "sub/d.txt")

    assert files(tmp_path) == {".gitignore", "a.py", "sub/b.py"}


def test_a_nested_file_wins_over_its_parents(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n")
    put(tmp_path, "keep.log", "sub/keep.log", "sub/other.log")
    (tmp_path / "sub" / ".gitignore").write_text("!keep.log\n")

    assert files(tmp_path) == {".gitignore", "sub/.gitignore", "sub/keep.log"}


def test_double_star(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("**/gen\nassets/**\na/**/z.txt\n")
    put(
        tmp_path,
        "gen/x",
        "deep/gen/x",
        "deep/ok",
        "assets/img.png",
        "assets/sub/img.png",
        "a/z.txt",
        "a/x/z.txt",
        "a/x/y/z.txt",
        "a/x/y.txt",
    )

    assert files(tmp_path) == {".gitignore", "deep/ok", "a/x/y.txt"}


def test_question_mark_and_classes(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "?.tmp\n[abc].dat\n[!x].out\n[0-9]*.bak\n"
    )
    put(
        tmp_path,
        "1.tmp",
        "12.tmp",
        "a.dat",
        "d.dat",
        "x.out",
        "y.out",
        "3old.bak",
        "old.bak",
    )

    assert files(tmp_path) == {
        ".gitignore",
        "12.tmp",
        "d.dat",
        "x.out",
        "old.bak",
    }


def test_comments_blank_lines_and_escapes(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "# a comment\n\n\\#literal\n\\!bang\nspaced\\ \nplain   \n"
    )
    put(
        tmp_path,
        "#literal",
        "!bang",
        "spaced ",
        "plain",
        "spaced",
        "# a comment",
    )

    assert files(tmp_path) == {".gitignore", "spaced", "# a comment"}


def test_the_enclosing_repository_governs_a_subdirectory(
    tmp_path: Path,
) -> None:
    """Copying `apps/web` out of a monorepo: the root's rules still hold,
    anchored where they were written, and so does `.git/info/exclude`."""
    (tmp_path / ".git" / "info").mkdir(parents=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("scratch/\n")
    (tmp_path / ".gitignore").write_text("*.secret\n/top-only\n")
    (tmp_path / "apps" / ".gitignore").parent.mkdir(parents=True)
    (tmp_path / "apps" / ".gitignore").write_text("*.local\n")
    web = tmp_path / "apps" / "web"
    put(web, "a.secret", "top-only", "cfg.local", "scratch/x", "index.ts")

    assert files(web) == {"top-only", "index.ts"}


def test_a_directory_that_is_not_a_repository_still_reads_its_file(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n")
    put(tmp_path, "a.log", "b.py")

    assert files(tmp_path) == {".gitignore", "b.py"}


# --- Against git itself -----------------------------------------------------

RULES = """# what a real project writes
*.log
!important.log
/dist
build/
node_modules
docs/drafts
**/gen
assets/**
a/**/z.txt
?.tmp
[abc].dat
[!x].out
\\#literal
spaced\\
*.secret
/top-only
"""

TREE = (
    "keep.py",
    "debug.log",
    "important.log",
    "sub/important.log",
    "sub/x.log",
    "dist/a.js",
    "src/dist/b.js",
    "build/o",
    "src/build",
    "src/build.py",
    "node_modules/m.js",
    "pkg/node_modules/m.js",
    "docs/drafts/d.md",
    "other/docs/drafts/d.md",
    "gen/x",
    "deep/gen/x",
    "deep/ok",
    "assets/i.png",
    "assets/sub/i.png",
    "a/z.txt",
    "a/x/z.txt",
    "a/x/y/z.txt",
    "a/x/y.txt",
    "1.tmp",
    "12.tmp",
    "a.dat",
    "d.dat",
    "x.out",
    "y.out",
    "#literal",
    "spaced ",
    "spaced",
    "apps/web/a.secret",
    "apps/web/top-only",
    "apps/web/cfg.local",
    "apps/web/index.ts",
    "apps/web/nested/.gitignore",
    "apps/web/nested/keep.log",
    "apps/web/nested/drop.log",
    "top-only",
)


def git_ls(cwd: Path) -> set[str]:
    """Every untracked file git would NOT ignore, relative to `cwd`.

    Global and system configuration are switched off so only the files in
    the tree speak — this must not depend on the developer's own excludes.
    """
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(cwd),
    }
    out = subprocess.run(
        [
            "git",
            "-c",
            f"core.excludesFile={os.devnull}",
            "ls-files",
            "-o",
            "--exclude-standard",
            "-z",
        ],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
    ).stdout.decode()
    return {p for p in out.split("\0") if p}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("needs git to compare against")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(RULES)
    put(tmp_path, *TREE)
    (tmp_path / "apps" / ".gitignore").write_text("*.local\n")
    (tmp_path / "apps" / "web" / "nested" / ".gitignore").write_text(
        "!keep.log\n"
    )
    return tmp_path


def test_reads_gitignore_the_way_git_does(repo: Path) -> None:
    assert files(repo, ignore=[".git"]) == git_ls(repo)


def test_reads_a_subdirectory_the_way_git_does(repo: Path) -> None:
    web = repo / "apps" / "web"

    assert files(web, ignore=[".git"]) == git_ls(web)
