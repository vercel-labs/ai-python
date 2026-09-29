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
from typing import Literal

from pydantic import BaseModel

from ai.harnesses.experimental import Handle

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

    @property
    def session_id(self) -> str:
        return self.handle.session_id


class State(BaseModel):
    """Everything afk remembers. Small on purpose."""

    remotes: list[Remote] = []
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
        self.remotes = [r for r in self.remotes if r.sandbox != name]
        self.baselines.pop(name, None)


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
