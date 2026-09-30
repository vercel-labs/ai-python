"""The verbs, each a thin composition of SDK calls.

push   — copy it into a sandbox and open the TUI there (or run it unattended).
attach — your terminal, back on its TUI; reopened there if it had exited.
peek   — watch an unattended agent's transcript, read-only.
pull   — bring a conversation home and open it here; --files brings its edits.
stop   — end its sandbox; each push has its own.

Nothing here depends on how a conversation began: `push` reads the
harness's own store, so a conversation started by typing `claude` in a
terminal is handled like any other.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai.harnesses.experimental.errors import HarnessError, SessionBusyError
from ai.workspaces.experimental import Local, VercelSandbox, copy, tty

from . import files, render
from . import state as st
from .rows import (
    CONTINUE,
    CONTINUE_MARK,
    HARNESSES,
    KIND_SHORT,
    Row,
    turn_finished,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ai.workspaces.experimental import Gateway


def _pty_name(label: str) -> str:
    return f"afk-{label}"


async def push(
    cwd: Path,
    row: Row,
    *,
    label: str | None,
    background: bool,
    hours: float,
    gateway: Gateway | None,
    local_gateway: Gateway | None = None,
) -> st.Remote:
    """Copy the conversation, and this directory as it is now, into a
    sandbox of its own. Push copies: the local one stays; if you keep typing
    here, the list shows both.

    Every push gets its own sandbox. A second push never lands in a tree an
    earlier one left behind, so there is nothing to catch up and no one
    else's agent in the same files; each sandbox ends with its conversation.

    `hours` is how long the sandbox may live. The
    platform's default is five MINUTES — measured: a pushed TUI was gone
    before its owner came back — so afk always says how long.

    `gateway` is how the sandbox reaches a model; `local_gateway` is only
    for this machine, where your own CLI login applies unless you set one.
    """
    if gateway is None:
        raise HarnessError(
            "afk push needs a way for the sandbox to reach a model: sign in "
            "to Vercel, or set AI_GATEWAY_API_KEY"
        )
    # 1. The past, from the harness's own store here.
    async with Local(cwd, gateway=local_gateway) as here:
        async with HARNESSES[row.kind](workspace=here) as agent:
            history = await agent.history(row.session_id)

    # 2. A sandbox of its own, with this directory as it is now.
    ws = VercelSandbox(
        keep=True,
        gateway=gateway,
        execution_time_limit=timedelta(hours=hours).total_seconds(),
    )
    await ws.open()
    recorded = False
    try:
        copied = await copy(cwd, ws / ".")
        _remember_baseline(
            ws.name or "", await asyncio.to_thread(files.manifest, cwd)
        )
        print(
            f"afk: sandbox {ws.name} · copied {copied} files · lives up to "
            f"{hours:g}h"
        )
        agent = HARNESSES[row.kind](workspace=ws)
        await agent.open()
        if background:
            session = agent.session(history=history)
            # A Turn is an async iterable: iterate it. Let the turn begin
            # before letting go — the first event proves the agent is
            # working — then this client detaches and the VM keeps it.
            events = aiter(session.stream(CONTINUE.format(path=ws.path)))
            first = asyncio.ensure_future(anext(events))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(first), 120)
            sid = session.session_id
            name = label or sid[:4]
            remote = st.Remote(
                label=name,
                origin=str(cwd),
                sandbox=ws.name or "",
                pty=None,
                mode="bg",
                handle=session.handle,
                from_id=row.session_id,
                pushed_at=st.now(),
            )
            _remember(remote)
            recorded = True
            first.cancel()
            with contextlib.suppress(BaseException):
                await first
            await agent.detach()
            print(f"afk: {name} ({sid[:8]}) · running unattended in {ws.name}")
            return remote
        tui = await agent.tui(
            history=history, name=_pty_name(label or "pending")
        )
        sid = tui.session_id or ""
        name = label or sid[:4]
        if tui.handle is None:
            raise HarnessError(
                "the harness did not report the new conversation's id"
            )
        remote = st.Remote(
            label=name,
            origin=str(cwd),
            sandbox=ws.name or "",
            pty=tui.pty.name,
            mode="tui",
            handle=tui.handle,
            from_id=row.session_id,
            pushed_at=st.now(),
        )
        _remember(remote)
        recorded = True
        print(
            f"afk: {name} ({sid[:8]}) · {ws.name} · attached — Ctrl-] detaches "
            "and leaves it running"
        )
        status = await tty.bridge(tui.pty)
        if status is None:
            print(f"afk: detached; `afk attach {name}` to come back")
            await agent.detach()
        else:
            print(f"afk: the TUI exited ({status})")
            await tui.close()
            await agent.close()
        return remote
    except BaseException:
        if not recorded and ws.name:
            # A sandbox nothing remembers is a sandbox nothing can stop: end
            # the one this push created before anything was recorded.
            with contextlib.suppress(Exception):
                await ws.close()
                async with VercelSandbox(
                    name=ws.name, keep=False, gateway=gateway
                ):
                    pass
            print(
                "afk: push failed; stopped the sandbox it had created "
                f"({ws.name})"
            )
            raise
        raise
    finally:
        await (
            ws.close()
        )  # keep=True: leaves the VM and everything in it running


async def attach(row: Row, gateway: Gateway | None) -> None:
    """Your terminal on the conversation's TUI in its sandbox.

    If the TUI is still running, reconnect to it; if it has exited (you quit it,
    or it crashed), open it again there on the same conversation.
    """
    if (
        row.mode != "tui"
        or not row.pty
        or not row.sandbox
        or row.handle is None
    ):
        raise HarnessError(
            f"{row.label or row.short} was not pushed as a TUI; `afk peek` "
            "watches an unattended one"
        )
    async with VercelSandbox(name=row.sandbox, gateway=gateway) as ws:
        live = {p.name: p for p in await ws.ptys()}
        if row.pty in live:
            if live[row.pty].attached:
                print(
                    f"afk: {row.label} is open in another terminal; taking it "
                    "over (that one returns)"
                )
            pty = await ws.attach(row.pty)
            print(
                f"afk: {row.label} · {row.sandbox} · attached — Ctrl-] detaches"
            )
            status = await tty.bridge(pty)
            if status is None:
                print(f"afk: detached; `afk attach {row.label}` to come back")
            else:
                print(
                    f"afk: the TUI exited ({status}); `afk attach {row.label}` "
                    "reopens it"
                )
            return
        agent = HARNESSES[row.kind](workspace=ws)
        await agent.open()
        try:
            tui = await agent.tui(row.session_id, name=row.pty)
        except SessionBusyError as busy:
            print(f"afk: {busy}")
            await agent.close()
            return
        print(
            f"afk: {row.label}'s TUI had exited; reopened it in {row.sandbox} "
            "— Ctrl-] detaches"
        )
        status = await tty.bridge(tui.pty)
        if status is None:
            print(f"afk: detached; `afk attach {row.label}` to come back")
            await agent.detach()
        else:
            print(
                f"afk: the TUI exited ({status}); `afk attach {row.label}` "
                "reopens it"
            )
            await tui.close()
            await agent.close()


PEEK_PAST = 4
"""Earlier messages peek shows before following along, at least; it starts
at the message that began the current turn when that is not far back."""

PEEK_TURN_WITHIN = 30


def _peek_start(messages: list[Any]) -> int:
    """Where to start drawing: your last message, so the turn reads whole
    from its prompt, unless that is far back; never on a tool result."""
    floor = max(0, len(messages) - PEEK_TURN_WITHIN)
    for i in range(len(messages) - 1, floor - 1, -1):
        m = messages[i]
        if m.role == "user" and any(
            getattr(p, "kind", "") == "text" for p in m.parts
        ):
            return min(i, max(0, len(messages) - PEEK_PAST))
    start = max(0, len(messages) - PEEK_PAST)
    while start < len(messages) and messages[start].role == "tool":
        start += 1
    return start


PEEK_HELD_EVERY = 5
"""Every how many polls peek asks whether the agent is still held at all."""

PEEK_QUIET = 8.0
"""Quiet seconds after the agent's last words before peek calls it done."""


async def peek(
    row: Row, gateway: Gateway | None, *, every: float = 2.0
) -> None:
    """Follow an unattended agent's transcript.

    Read-only: it reads the stored record, so it never becomes a second writer.
    Returns when the agent has finished its turn; Ctrl-C stops watching, never
    the agent.
    """
    if row.handle is None or not row.sandbox:
        raise HarnessError("nothing to peek at")
    width = shutil.get_terminal_size().columns
    style = render.Style(color=sys.stdout.isatty())
    name = row.label or row.short
    async with VercelSandbox(name=row.sandbox, gateway=gateway) as ws:
        async with HARNESSES[row.kind](workspace=ws) as agent:
            print(
                style.dim(
                    f"afk: watching {name} · {KIND_SHORT[row.kind]} · "
                    f"{row.sandbox} — Ctrl-C stops watching; the agent keeps "
                    "going"
                )
            )
            messages = await agent.history(row.session_id)
            shown = _peek_start(messages)
            if shown:
                print(
                    style.dim(
                        f"  … {shown} earlier message{'s' * (shown != 1)}"
                    )
                )
            loop = asyncio.get_running_loop()
            changed = loop.time()
            polls = 0
            while True:
                for m in messages[shown:]:
                    drawn = render.lines(m, width, style)
                    if drawn and m.role == "user":
                        print()
                    for line in drawn:
                        print(line)
                if len(messages) > shown:
                    changed = loop.time()
                shown = len(messages)
                # The finished turn ends a peek; this check only catches an
                # agent that died, so it need not list every session each poll.
                polls += 1
                held = polls % PEEK_HELD_EVERY != 1 or any(
                    i.running
                    for i in await agent.sessions()
                    if i.session_id == row.session_id
                )
                if not held:
                    print(
                        style.dim(
                            f"\nafk: {name} is no longer running · `afk pull "
                            f"{name} --files` brings it home"
                        )
                    )
                    return
                if (
                    turn_finished(
                        messages, CONTINUE_MARK if row.mode == "bg" else None
                    )
                    and loop.time() - changed >= PEEK_QUIET
                ):
                    print(
                        style.dim(
                            f"\nafk: {name} finished its turn · `afk pull "
                            f"{name} --files` brings it home"
                        )
                    )
                    return
                await asyncio.sleep(every)
                messages = await agent.history(row.session_id)


async def pull(
    cwd: Path,
    row: Row,
    gateway: Gateway | None,
    *,
    with_files: bool,
    local_gateway: Gateway | None = None,
    ask: Callable[[str], str] = input,
) -> None:
    """Bring it home: the conversation's past, into a new one here, in the
    TUI. With `with_files`, the files it changed in the sandbox too — shown
    first, and written only on your yes."""
    if row.handle is None or not row.sandbox:
        raise HarnessError("nothing to pull")
    async with VercelSandbox(name=row.sandbox, gateway=gateway) as ws:
        async with HARNESSES[row.kind](workspace=ws) as there:
            history = await there.history(row.session_id)
        if with_files:
            await _pull_files(
                cwd, ws, st.load().baselines.get(row.sandbox), ask
            )
    async with Local(cwd, gateway=local_gateway) as here:
        agent = HARNESSES[row.kind](workspace=here)
        await agent.open()
        # An ordinary local TUI: no detach key, quit it as you always do.
        tui = await agent.tui(history=history)
        print(
            f"afk: {row.label} · home as {(tui.session_id or '')[:8]} — quit "
            "the TUI as usual when done"
        )
        await tty.bridge(tui.pty, detach_key=None)
        await tui.close()
        await agent.close()
        print(
            "afk: it is in this directory's list now, like any conversation "
            "here"
        )


async def _pull_files(
    cwd: Path,
    ws: VercelSandbox,
    baseline: dict[str, str] | None,
    ask: Callable[[str], str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="afk-pull-") as staging:
        await copy(ws / ".", staging)
        there = await asyncio.to_thread(files.manifest, Path(staging))
        here = await asyncio.to_thread(files.manifest, cwd)
        changes = files.plan(there, here, baseline)
        if not changes:
            print("afk: the sandbox changed no files; nothing to bring home")
            return
        clean = [c for c in changes if not c.conflict]
        conflicts = [c for c in changes if c.conflict]
        if baseline is None:
            print(
                "afk: this sandbox was pushed before afk recorded what it "
                "copied, so its changes and"
            )
            print(
                "     yours cannot be told apart; every file that differs is "
                "listed, and none is deleted"
            )
        if clean:
            print(
                f"afk: the sandbox changed {len(clean)} "
                f"file{'s' * (len(clean) != 1)}:"
            )
            for c in clean:
                print(f"       {c.action:7s} {c.path}")
        if conflicts:
            what = (
                "differ"
                if baseline is None
                else "changed here too since the push"
            )
            print(
                f"afk: {len(conflicts)} file{'s' * (len(conflicts) != 1)} "
                f"{what}; writing would lose your local version:"
            )
            for c in conflicts:
                print(f"       {c.action:7s} {c.path}")
        chosen = []
        if clean and _yes(
            ask,
            f"afk: write the {len(clean)} change{'s' * (len(clean) != 1)} "
            "here? [y/N] ",
        ):
            chosen += clean
        if conflicts and _yes(
            ask,
            f"afk: overwrite your version of {len(conflicts)} "
            f"file{'s' * (len(conflicts) != 1)}? [y/N] ",
        ):
            chosen += conflicts
        for c in chosen:
            target = cwd / c.path
            if c.action == "delete":
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(staging) / c.path, target)
        kept = len(changes) - len(chosen)
        print(
            f"afk: wrote {len(chosen)} file{'s' * (len(chosen) != 1)}"
            + (f"; left {kept} as they are" if kept else "")
        )


def _yes(ask: Callable[[str], str], prompt: str) -> bool:
    try:
        return ask(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


async def stop(row: Row, gateway: Gateway | None) -> None:
    """End the conversation's sandbox — the only reliable way to end what
    runs in it. Each push has its own, so nothing else goes with it."""
    if not row.sandbox:
        raise HarnessError("nothing to stop")
    # The platform's stop takes a few seconds and cannot be hurried: say so
    # first.
    print(f"afk: stopping {row.label} ({row.sandbox})…", flush=True)
    async with VercelSandbox(name=row.sandbox, keep=False, gateway=gateway):
        pass
    current = st.load()
    # A sandbox shared by several pushes predates one-sandbox-per-push; say so.
    also = [
        r.label
        for r in current.remotes
        if r.sandbox == row.sandbox and r.session_id != row.session_id
    ]
    current.forget_sandbox(row.sandbox)
    st.save(current)
    print(
        f"afk: stopped {row.label} ({row.sandbox})"
        + (f"; it also held {', '.join(also)}" if also else "")
    )


def _remember_baseline(sandbox: str, manifest: dict[str, str]) -> None:
    current = st.load()
    current.baselines[sandbox] = manifest
    st.save(current)


def _remember(remote: st.Remote) -> None:
    current = st.load()
    current.remotes = [
        r for r in current.remotes if r.session_id != remote.session_id
    ]
    current.remotes.append(remote)
    st.save(current)
