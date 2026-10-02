"""The list: every conversation here, and every one afk pushed elsewhere.

`here` comes straight from the harnesses' own stores — nothing was
registered, so a conversation you started by typing `claude` an hour ago is
listed like any other. `remote` comes from afk's state and is then *verified*
against the sandbox: does the VM still exist, is the TUI's pty still there,
is the conversation still held. The machine is the source of truth; the
state file only holds what cannot be asked.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ai.harnesses.experimental import Handle, claude_code, codex
from ai.harnesses.experimental.errors import HarnessError
from ai.workspaces.experimental import Local, VercelSandbox
from ai.workspaces.experimental.errors import (
    WorkspaceError,
    WorkspaceGoneError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ai.workspaces.experimental import Gateway

    from .state import Remote, State

HARNESSES = {"claude-code": claude_code, "codex": codex}
KIND_SHORT = {"claude-code": "claude", "codex": "codex"}

CONTINUE_MARK = "This conversation was moved here from another machine."
CONTINUE = (
    CONTINUE_MARK
    + " The project now lives "
    + "at {path}. Continue the work from where it left off."
)
"""What an unattended push says to the agent.

afk wrote it, so finding it in a transcript is how afk knows where the pushed
turn began.
"""
ACTIVE_WITHIN = 10 * 60
"""A conversation written to this recently is one you are probably in."""

Where = Literal["here", "remote"]


class Row(BaseModel):
    """One conversation as the list shows it."""

    where: Where
    kind: str
    session_id: str
    title: str | None = None
    updated_at: int | None = None
    status: str
    label: str | None = None
    sandbox: str | None = None
    pty: str | None = None
    from_id: str | None = None
    handle: Handle | None = None
    mode: str | None = None
    running: bool = False
    """Here: a process has it open, maybe in another terminal."""
    pid: int | None = None
    """Here: that process, when the harness can tell."""

    @property
    def short(self) -> str:
        return self.session_id[:4]

    @property
    def active(self) -> bool:
        return (
            self.updated_at is not None
            and time.time() - seconds(self.updated_at) < ACTIVE_WITHIN
        )

    def matches(self, word: str) -> bool:
        w = word.lower()
        return (
            self.session_id.lower().startswith(w)
            or (self.label or "").lower() == w
            or w in (self.title or "").lower()
        )


def seconds(ts: int) -> float:
    """Timestamps arrive in whatever unit a harness's store uses — claude's
    are milliseconds, codex's seconds. Anything past the year 33658 is ms."""
    return ts / 1000 if ts > 10**11 else float(ts)


def age(ts: int | None) -> str:
    if ts is None:
        return "?"
    s = max(0, int(time.time() - seconds(ts)))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= size:
            return f"{s // size}{unit}"
    return f"{s}s"


async def local_rows(
    cwd: Path, gateway: Gateway | None
) -> tuple[list[Row], list[str], list[str]]:
    """This directory's conversations, both harnesses.

    Returns rows, notes (a harness that is not installed here is a note, not an
    error), and the harnesses that answered.
    """
    rows: list[Row] = []
    notes: list[str] = []
    answered: list[str] = []

    async def ask(factory: Any) -> Any:
        try:
            async with Local(cwd, gateway=gateway) as ws:
                async with factory(workspace=ws) as agent:
                    return await agent.sessions()
        except HarnessError as exc:
            return exc

    # Both harnesses at once: neither waits on the other's store.
    results = await asyncio.gather(*(ask(f) for f in HARNESSES.values()))
    for kind, infos in zip(HARNESSES, results, strict=True):
        if isinstance(infos, HarnessError):
            notes.append(
                f"{KIND_SHORT[kind]}: {str(infos).splitlines()[0][:80]}"
            )
            continue
        answered.append(KIND_SHORT[kind])
        for info in infos:
            row = Row(
                where="here",
                kind=kind,
                session_id=info.session_id,
                title=info.title,
                updated_at=info.updated_at,
                status="",
                running=info.running,
                pid=info.pid,
            )
            row.status = (
                "active " + age(info.updated_at)
                if row.active
                else "idle " + age(info.updated_at)
            )
            if info.running:
                row.status = "in use " + age(info.updated_at)
            rows.append(row)
    rows.sort(key=lambda r: seconds(r.updated_at or 0), reverse=True)
    return rows, notes, answered


async def remote_rows(
    state: State, origin: Path, gateway: Gateway | None
) -> tuple[list[Row], list[str]]:
    """Conversations afk pushed from this directory, verified against their
    sandboxes. A sandbox the platform says no longer exists marks its rows
    `gone` and is forgotten after this listing; one that cannot be reached
    right now is `unreachable` and kept."""
    rows: list[Row] = []
    gone: list[str] = []
    remotes = state.for_origin(str(origin))
    by_sandbox: dict[str, list[Remote]] = {}
    for r in remotes:
        by_sandbox.setdefault(r.sandbox, []).append(r)

    async def probe(name: str, group: list[Remote]) -> None:
        try:
            async with VercelSandbox(name=name, gateway=gateway) as ws:
                # Ask only what the rows need: a TUI's status is its pty, so
                # a sandbox holding only TUIs never opens a harness (measured:
                # the harness open and session listing were ~3s of a 4s probe).
                if any(r.mode == "tui" for r in group):
                    ptys = {p.name: p for p in await ws.ptys()}
                else:
                    ptys = {}
                for r in group:
                    if r.mode == "tui":
                        # Running with a terminal on it (yours, in another
                        # tab, or someone's) vs running with nobody watching.
                        info = ptys.get(r.pty or "")
                        status = (
                            "idle"
                            if info is None
                            else "open elsewhere"
                            if info.attached
                            else "running"
                        )
                    else:
                        status = await _unattended_status(ws, r)
                    rows.append(_remote_row(r, status))
        except WorkspaceGoneError:
            # The platform says it is stopped or was never there: forget it.
            gone.append(name)
            for r in group:
                rows.append(_remote_row(r, "gone"))
        except (HarnessError, WorkspaceError):
            # Could not ask — network, credentials, the platform (the SDK says
            # so as a WorkspaceError). It may well be running: keep the
            # record, or it is the only way back lost.
            for r in group:
                rows.append(_remote_row(r, "unreachable"))

    await asyncio.gather(*(probe(n, g) for n, g in by_sandbox.items()))
    rows.sort(key=lambda r: seconds(r.updated_at or 0), reverse=True)
    return rows, gone


async def _unattended_status(ws: VercelSandbox, r: Remote) -> str:
    """running, finished, or idle — from one harness, opened once."""
    try:
        async with HARNESSES[r.handle.kind](workspace=ws) as agent:
            held = any(
                i.running
                for i in await agent.sessions()
                if i.session_id == r.session_id
            )
            if not held:
                return "idle"
            # An unattended agent stays held after its turn, waiting for a
            # next one that will not come: say which it is.
            return (
                "finished"
                if turn_finished(
                    await agent.history(r.session_id), CONTINUE_MARK
                )
                else "running"
            )
    except (HarnessError, WorkspaceError):
        return "?"


def stored_rows(state: State, origin: Path) -> list[Row]:
    """The conversations afk pushed from here, from its own record alone —
    no sandbox asked. What a verb needs to find the one you named: attach,
    peek, pull and stop then talk to that one sandbox, and nothing else."""
    rows = [_remote_row(r, "") for r in state.for_origin(str(origin))]
    rows.sort(key=lambda r: seconds(r.updated_at or 0), reverse=True)
    return rows


def _remote_row(r: Remote, status: str) -> Row:
    return Row(
        where="remote",
        kind=r.handle.kind,
        session_id=r.session_id,
        title=None,
        updated_at=r.pushed_at,
        status=status,
        label=r.label,
        sandbox=r.sandbox,
        pty=r.pty,
        from_id=r.from_id,
        handle=r.handle,
        mode=r.mode,
    )


def turn_finished(messages: list[Any], asked: str | None = None) -> bool:
    """Whether the transcript ends on the agent's own words with no tool call
    pending: its turn is over and it waits for the next one. A turn in
    progress ends on a tool call or a tool result.

    `asked` is text from the prompt that began the turn in question. Until
    that prompt is in the transcript, the turn has not shown up yet and the
    reply before it proves nothing. (Measured: message counts differ between
    the history sent and the transcript stored, so a count cannot say it.)"""
    if asked is not None:
        begun = [
            i
            for i, m in enumerate(messages)
            if m.role == "user" and asked in _text(m)
        ]
        if not begun or begun[-1] == len(messages) - 1:
            return False
    if not messages or messages[-1].role != "assistant":
        return False
    kinds = {getattr(p, "kind", "") for p in messages[-1].parts}
    return "text" in kinds and "tool_call" not in kinds


def _text(message: Any) -> str:
    return " ".join(
        getattr(p, "text", "") or ""
        for p in message.parts
        if getattr(p, "kind", "") == "text"
    )
