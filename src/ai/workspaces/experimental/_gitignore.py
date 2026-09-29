"""What a project's `.gitignore` says not to ship.

`copy()` honours it by default. The files a project keeps out of version
control — `.env`, build output, editor state — are the files nobody means to
hand an agent, and the project has already written down which they are.

This is a reading of gitignore(5) in Python rather than a call to `git`: it
works on a directory that is not a repository, on a machine without git, and
on a tree that arrived in an archive. Read are every `.gitignore` under the
directory being copied, the ones above it up to the root of the enclosing
repository (a subdirectory of a monorepo is governed by the root's rules
too), and `.git/info/exclude`. The global excludes file is not: it describes
this machine's habits, not the project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

GITIGNORE = ".gitignore"


@dataclass(frozen=True)
class Rule:
    """One line of an ignore file, compiled."""

    regex: re.Pattern[str]
    negate: bool
    dir_only: bool
    anchored: bool
    """A pattern with a slash in it is matched against the path relative to
    its file's directory; one without is matched against the entry's name,
    at any depth."""

    def matches(self, relative: str, name: str, *, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        return (
            self.regex.fullmatch(relative if self.anchored else name)
            is not None
        )


@dataclass(frozen=True)
class Scope:
    """The rules of one ignore file, and the directory they are relative to."""

    base: str
    """Return that directory as a posix prefix: "" or ending in "/"."""
    rules: tuple[Rule, ...]


def parse(text: str) -> tuple[Rule, ...]:
    """Compile the lines of one ignore file, in order."""
    rules: list[Rule] = []
    for raw in text.splitlines():
        line = _trim(raw)
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        anchored = "/" in line
        line = line.lstrip("/")
        if not line:
            continue
        rules.append(
            Rule(re.compile(_translate(line)), negate, dir_only, anchored)
        )
    return tuple(rules)


def _trim(line: str) -> str:
    """Trailing spaces are ignored unless a backslash keeps them."""
    stripped = line.rstrip(" ")
    if stripped.endswith("\\") and len(stripped) < len(line):
        return stripped + " "
    return stripped


def _translate(glob: str) -> str:
    """Translate a gitignore glob to a regular expression over a path.

    Wildcards never cross a slash; `**` is the one that does, and only as a
    whole path segment.
    """
    out: list[str] = []
    i, n = 0, len(glob)
    while i < n:
        c = glob[i]
        if c == "*":
            j = i
            while j < n and glob[j] == "*":
                j += 1
            whole_segment = (i == 0 or glob[i - 1] == "/") and (
                j == n or glob[j] == "/"
            )
            if j - i >= 2 and whole_segment:
                if j == n:
                    out.append(".*")  # trailing "/**": everything inside
                else:
                    out.append("(?:.*/)?")  # "**/": zero or more directories
                    j += 1
            else:
                out.append("[^/]*")
            i = j
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            end = _class_end(glob, i)
            if end is None:
                out.append(re.escape(c))
                i += 1
            else:
                body = glob[i + 1 : end]
                if body[0] in "!^":
                    body = "^" + body[1:]
                out.append(
                    "[" + body.replace("\\", "\\\\").replace("[", "\\[") + "]"
                )
                i = end + 1
        elif c == "\\" and i + 1 < n:
            out.append(re.escape(glob[i + 1]))
            i += 2
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _class_end(glob: str, start: int) -> int | None:
    """Index of the `]` closing the class opened at `start`, if it is one."""
    j = start + 1
    if j < len(glob) and glob[j] in "!^":
        j += 1
    if j < len(glob) and glob[j] == "]":
        j += 1  # a leading "]" is a literal member
    while j < len(glob) and glob[j] != "]":
        j += 1
    return j if j < len(glob) else None


def _read(file: Path) -> tuple[Rule, ...]:
    return parse(file.read_text(errors="replace"))


class GitIgnore:
    """The ignore rules that govern one directory tree.

    Built once per copy. The walk asks it about every entry, and tells it
    when it descends into a directory so that directory's own file joins the
    rules in force — deeper files come later, so they win.
    """

    def __init__(self, source: Path) -> None:
        source = source.resolve()
        top = next(
            (d for d in (source, *source.parents) if (d / ".git").exists()),
            source,
        )
        self._prefix = (
            "" if source == top else source.relative_to(top).as_posix() + "/"
        )

        scopes: list[Scope] = []
        exclude = top / ".git" / "info" / "exclude"
        if (top / ".git").is_dir() and exclude.is_file():
            scopes.append(Scope("", _read(exclude)))
        # The files above the directory named, from the repository root down
        # to its parent. Its own is picked up by the walk, like any other.
        above: list[Path] = []
        if source != top:
            above.append(top)
            for part in source.relative_to(top).parts[:-1]:
                above.append(above[-1] / part)
        for directory in above:
            file = directory / GITIGNORE
            if file.is_file():
                base = (
                    ""
                    if directory == top
                    else directory.relative_to(top).as_posix() + "/"
                )
                scopes.append(Scope(base, _read(file)))
        self.above: tuple[Scope, ...] = tuple(scopes)

    def descend(
        self, scopes: tuple[Scope, ...], directory: Path, prefix: str
    ) -> tuple[Scope, ...]:
        """Return the scopes in force inside `directory`.

        Its path relative to the tree being copied is `prefix` ("" or ending in
        "/").
        """
        file = directory / GITIGNORE
        if not file.is_file():
            return scopes
        return (*scopes, Scope(self._prefix + prefix, _read(file)))

    def ignored(
        self,
        scopes: tuple[Scope, ...],
        relative: str,
        name: str,
        *,
        is_dir: bool,
    ) -> bool:
        """Whether the last matching rule says so."""
        path = self._prefix + relative
        verdict = False
        for scope in scopes:
            within = path[len(scope.base) :]
            for rule in scope.rules:
                if rule.matches(within, name, is_dir=is_dir):
                    verdict = not rule.negate
        return verdict
