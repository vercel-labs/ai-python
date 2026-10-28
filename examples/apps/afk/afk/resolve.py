"""Turning what you typed into one conversation — or asking, never guessing.

Humans mostly do not type ids. A verb accepts a label you gave, a word
from a title, or a short id prefix, whatever the list showed. With no
argument, the one conversation clearly in use is chosen; with several in
play, afk shows them and asks. The only thing ever auto-selected is a
choice with exactly one candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .rows import Row


class AmbiguousError(Exception):
    def __init__(self, query: str, matches: list[Row]) -> None:
        self.query = query
        self.matches = matches
        super().__init__(f"{len(matches)} conversations match {query!r}")


class NoMatchError(Exception):
    pass


def match(
    rows: list[Row], query: str | None, *, where: str | None = None
) -> Row:
    """Resolve `query` against rows.

    Raises AmbiguousError or NoMatchError; the CLI turns those into a
    numbered pick or a message.
    """
    pool = [r for r in rows if where is None or r.where == where]
    if query is None:
        current = [r for r in pool if r.where == "here" and r.active]
        if len(current) == 1:
            return current[0]
        raise AmbiguousError(
            "(the current conversation)", pool if len(current) != 1 else current
        )
    exact = [
        r for r in pool if r.session_id == query or (r.label or "") == query
    ]
    if len(exact) == 1:
        return exact[0]
    found = [r for r in pool if r.matches(query)]
    if len(found) == 1:
        return found[0]
    if not found:
        raise NoMatchError(query)
    raise AmbiguousError(query, found)


def pick(
    rows: list[Row], prompt: str, ask: Callable[[str], str] = input
) -> Row | None:
    """Numbered choice on this screen. Returns None when the user declines."""
    if not rows:
        return None
    for n, r in enumerate(rows, 1):
        title = (r.title or r.label or "").strip()[:48]
        print(
            f"  {n:>2}. {r.short}  {r.kind.split('-')[0]:6s} {title:48s} "
            f"{r.status}"
        )
    while True:
        answer = ask(f"{prompt} [1-{len(rows)}, Enter to cancel]: ").strip()
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(rows):
            return rows[int(answer) - 1]
        print("  a number from the list, please")
