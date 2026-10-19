"""What only afk knows: which conversations it pushed where, and from where.

The SDK asks the machine for everything it can — which conversations a
harness has in a directory, which ptys a workspace still runs, whether a
conversation is held. What exists only because *you* did something is
recorded here: the sandbox a push created, the pty name a TUI runs under,
the directory the conversation came from, the label you gave it, and the
`Handle` to come back with. One file, written once per push, verified
against the machine every time it is read.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ai.harnesses.experimental import Handle

from .gitflow import GitPush

CONTINUE_MARK = "This conversation was moved here from another machine."
"""The start of the continue prompt (see rows.CONTINUE)."""

Mode = Literal["tui", "bg"]


class Remote(BaseModel):
    """A conversation afk pushed into a sandbox."""

    label: str
    origin: str
    """The local directory it was pushed from — the app's concept; the
    transcript in the VM only knows /vercel/sandbox."""
    sandbox: str
    pty: str | None = None
    """The named pty its TUI runs under, when pushed as a TUI."""
    mode: Mode
    handle: Handle
    from_id: str | None = None
    """The local conversation it was copied from. Push copies; both remain."""
    pushed_at: int
    git: GitPush | None = None
    """In a repository: the branch the work is on, to bring it home."""
    asked: str | None = None
    """What an unattended push said to the agent: `--prompt`, or the
    continue prompt. Finding it in the transcript is how afk knows where
    the pushed turn began. None in records from before it was kept."""

    @property
    def mark(self) -> str:
        return self.asked or CONTINUE_MARK

    @property
    def session_id(self) -> str:
        return self.handle.session_id


class State(BaseModel):
    """Everything afk remembers. Small on purpose."""

    remotes: list[Remote] = []
    seen: dict[str, str] = {}
    """Per pushed conversation: the status the list last showed you, to say
    what changed since."""
    told_billing: bool = False
    """Whether the first push has said how sandboxes and models are billed."""
    baselines: dict[str, dict[str, str]] = {}
    """Per sandbox: sha256 of every file push copied into it, by path. What
    `pull --files` compares against to tell the sandbox's changes from yours."""

    def for_origin(self, origin: str) -> list[Remote]:
        return [r for r in self.remotes if r.origin == origin]

    def sandboxes(self, origin: str | None = None) -> list[str]:
        seen: list[str] = []
        for r in self.remotes:
            if (origin is None or r.origin == origin) and r.sandbox not in seen:
                seen.append(r.sandbox)
        return seen

    def forget_sandbox(self, name: str) -> None:
        for r in self.remotes:
            if r.sandbox == name:
                self.seen.pop(r.session_id, None)
        self.remotes = [r for r in self.remotes if r.sandbox != name]
        self.baselines.pop(name, None)

    def changed(self, rows: list[Any]) -> list[tuple[Any, str]]:
        """The rows whose status changed since the list last showed them,
        with what it was; and remember what it is now."""
        out = []
        for row in rows:
            before = self.seen.get(row.session_id)
            if before is not None and before != row.status and row.status:
                out.append((row, before))
            if row.status and row.status != "unreachable":
                self.seen[row.session_id] = row.status
        return out


def state_path() -> Path:
    """`~/.afk/state.json`, or wherever AFK_STATE points (tests)."""
    override = os.environ.get("AFK_STATE")
    return Path(override) if override else Path.home() / ".afk" / "state.json"


def load() -> State:
    path = state_path()
    if not path.exists():
        return State()
    return State.model_validate_json(path.read_text())


def save(state: State) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))


def now() -> int:
    return int(time.time())
