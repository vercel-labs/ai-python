"""afk — step away; your agents don't.

    afk                      what's here, what's still running elsewhere
    afk push [id] [--bg]     into a sandbox: the TUI there, or unattended
    afk attach <id>          your terminal, back on its TUI (reopened if it
                             exited)
    afk peek <id>            watch a conversation, read-only, here or pushed
    afk pull <id> [--files]  bring it home, in the TUI; --files brings its
                             edits too
    afk stop <id>            stop its sandbox, keeping its files to resume;
                             here, end the process it is open in
    afk rm <id>              delete its sandbox and snapshot
    afk wait [id]            poll until it (or any pushed one) stops
                             running, then ring the terminal bell
    afk setup                choose the Vercel team afk's sandboxes use

An id is whatever the list showed: a label you gave (`--as`), a word from a
title, or a short id prefix. With none, the one conversation clearly in use
is chosen; with several in play, afk asks. Ctrl-] detaches from any TUI.

On a terminal, bare `afk` is a picker: arrow keys choose a conversation,
Enter or a letter runs the command shown for it, q leaves the list as is.
Enter puts you in the conversation: its harness's own resume for one here,
attach for a pushed one, or a peek at one something else is driving.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ai.harnesses.experimental.errors import HarnessError
from ai.workspaces.experimental import Gateway, vercel_ai_gateway
from ai.workspaces.experimental.errors import (
    NotAuthenticatedError,
    WorkspaceError,
    WorkspaceGoneError,
)

from . import account, verbs
from . import state as st
from .resolve import AmbiguousError, NoMatchError, match, pick
from .rows import (
    KIND_SHORT,
    Row,
    display_name,
    local_rows,
    remote_rows,
    stored_rows,
    titled,
)

if TYPE_CHECKING:
    from typing import TextIO

DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"


def gateway() -> Gateway | None:
    """This machine's gateway: only when you set AI_GATEWAY_API_KEY.

    Otherwise the CLIs here use your own login, as they do without afk.
    """
    return vercel_ai_gateway() if os.environ.get("AI_GATEWAY_API_KEY") else None


def sandbox_gateway(*, fresh: bool = False) -> Gateway | None:
    """How a sandbox reaches a model, signing in to Vercel on first use.

    AI_GATEWAY_API_KEY when set; otherwise the project token afk mints from
    your `vercel login`, which AI Gateway accepts too.
    """
    own = gateway()
    token = account.ensure(
        interactive=account.interactive(),
        fresh=fresh,
        # Only a push fixes a sandbox's model credential, so only a push
        # checks that the team's AI Gateway will serve afk's token.
        for_gateway=fresh and own is None,
    )
    return own or (vercel_ai_gateway(api_key=token) if token else None)


LIST_CAP = 15
DEFAULT_HOURS = 5.0
"""A pushed sandbox's lifetime.

The platform's own default is five minutes, which is shorter than any time away;
five hours is its maximum on Pro.
"""


async def listing(
    cwd: Path,
    gw: Gateway | None,
    remote_gw: Gateway | None,
    *,
    everything: bool,
) -> None:
    """The list, drawn as it arrives: this directory's conversations as soon
    as the local stores answer, the pushed ones once their sandboxes do —
    a slow sandbox never holds up what is already known."""
    state = st.load()
    away = (
        asyncio.create_task(remote_rows(state, cwd, remote_gw))
        if state.for_origin(str(cwd))
        else None
    )
    here, notes, answered = await local_rows(cwd, gw)
    print(
        f"{BOLD}{_tilde(cwd)}{RESET}  {DIM}·  "
        f"{', '.join(answered) or 'no harness found'}{RESET}"
    )
    for note in notes:
        print(f"  {DIM}{note}{RESET}")
    if not here and away is None:
        print()
        print(
            "  no conversations here yet — start one with `claude`, `codex` "
            + "or `pi`"
        )
        print()
        return
    if here:
        print(f"\n  {BOLD}here{RESET}")
        shown = here if everything else here[:LIST_CAP]
        for r in shown:
            print(
                f"  {r.short}  {KIND_SHORT[r.kind]:6s} {_title(r):46s} "
                f"{r.status}"
            )
        if len(here) > len(shown):
            print(
                f"  {DIM}… {len(here) - len(shown)} more; `afk --all` shows "
                f"them{RESET}"
            )
    if away is not None:
        print(f"\n  {BOLD}remote{RESET}", flush=True)
        remote, gone = await away
        remote = titled(remote, here)
        news = {r.session_id: was for r, was in state.changed(remote)}
        for r in remote:
            origin = f"← from {r.from_id[:4]} · " if r.from_id else ""
            detail = f"{r.detail} · " if r.detail else ""
            new = (
                f"{BOLD}← new, was {news[r.session_id]}{RESET} "
                if r.session_id in news
                else ""
            )
            called = _clip(display_name(r), 34)
            print(
                f"  {r.short}  {KIND_SHORT[r.kind]:6s} {called:34s} "
                f"{r.status:9s} {(r.cost or ''):18s} "
                f"{new}{DIM}{detail}{origin}{r.sandbox}{RESET}"
            )
        st.save(state)
        if any("/h" in (r.cost or "") for r in remote):
            print(
                f"  {DIM}/h: memory, every hour it runs; CPU adds while it "
                f"computes{RESET}"
            )
        if gone:
            for name in gone:
                state.forget_sandbox(name)
            st.save(state)
    print(
        f"\n{DIM}afk push <id> [--bg] · attach <id> · peek <id> · pull <id> "
        f"[--files] · stop <id> · rm <id>{RESET}"
    )


async def picking(
    cwd: Path,
    gw: Gateway | None,
    remote_gw: Gateway | None,
    *,
    everything: bool,
) -> list[list[str]] | None:
    """The list as a picker, on a terminal: the commands you chose, to run as
    if typed, or None. The same rows as `listing`, arriving the same way."""
    # textual is imported only when a picker is drawn
    from rich.text import Text  # noqa: PLC0415

    from . import tui  # noqa: PLC0415

    state = st.load()

    local = asyncio.create_task(local_rows(cwd, gw))

    async def away() -> list[Row]:
        remote, gone = await remote_rows(state, cwd, remote_gw)
        remote = titled(remote, (await local)[0])
        for name in gone:
            state.forget_sandbox(name)
        for r, was in state.changed(remote):
            r.detail = f"new, was {was}" + (
                f" · {r.detail}" if r.detail else ""
            )
        st.save(state)
        return remote

    probing = (
        asyncio.create_task(away()) if state.for_origin(str(cwd)) else None
    )
    here, notes, answered = await local
    where = f"{BOLD}{_tilde(cwd)}{RESET}  {DIM}·  "
    harnesses = ", ".join(answered) or "no harness found"
    if not here and probing is None:
        print(f"{where}{harnesses}{RESET}")
        for note in notes:
            print(f"  {DIM}{note}{RESET}")
        print(
            "\n  no conversations here yet — start one with `claude`, "
            "`codex` or `pi`\n"
        )
        return None
    shown = here if everything else here[:LIST_CAP]
    if len(here) > len(shown):
        notes.append(f"… {len(here) - len(shown)} more; `afk --all` shows them")
    head = Text.assemble((_tilde(cwd), "bold"), (f"  ·  {harnesses}", "dim"))
    for note in notes:
        head.append(f"\n  {note}", style="dim")
    return await tui.pick(head, shown, probing)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _title(r: Row) -> str:
    t = (r.title or "").strip().replace("\n", " ")
    return (t[:43] + "…") if len(t) > 44 else t


def _tilde(p: Path) -> str:
    home = str(Path.home())
    s = str(p)
    return "~" + s[len(home) :] if s.startswith(home) else s


def choose(
    rows: list[Row], query: str | None, *, where: str | None, verb: str
) -> Row | None:
    try:
        return match(rows, query, where=where)
    except NoMatchError:
        print(f"afk: nothing matches {query!r}; `afk` lists what there is")
        return None
    except AmbiguousError as amb:
        print(f"afk {verb}: which one?")
        return pick(amb.matches, f"afk {verb}")


async def main_async(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="afk",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        dest="everything",
        action="store_true",
        help="list every conversation, not just the recent ones",
    )
    sub = parser.add_subparsers(dest="verb")
    p_push = sub.add_parser("push", help="into a sandbox")
    p_push.add_argument("id", nargs="?")
    p_push.add_argument(
        "--bg",
        action="store_true",
        help="run unattended instead of opening the TUI",
    )
    p_push.add_argument("--as", dest="label", help="a label to call it by")
    p_push.add_argument(
        "--git-history",
        action="store_true",
        help=(
            "in a git repository, give the sandbox its whole history (for "
            "tasks that read `git log`), not only the tree"
        ),
    )
    p_push.add_argument(
        "--no-network",
        dest="network",
        action="store_false",
        help=(
            "lock the sandbox's network: only the model gateway (and npm, "
            "to install the CLI) are reachable"
        ),
    )
    p_push.add_argument(
        "--prompt",
        help=(
            "with --bg: what to tell the agent, instead of asking it to "
            "continue from where it left off"
        ),
    )
    p_push.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help=(
            "how long a sandbox this push creates may live (default "
            f"{DEFAULT_HOURS:g}; Hobby plans allow 0.75)"
        ),
    )
    for verb in ("attach", "peek"):
        sub.add_parser(verb).add_argument("id")
    p_pull = sub.add_parser("pull")
    p_pull.add_argument("id")
    p_pull.add_argument(
        "--files",
        action="store_true",
        help="also bring home the files it changed there; asks first",
    )
    sub.add_parser(
        "stop", help="stop its sandbox; its files are kept to resume"
    ).add_argument("id")
    sub.add_parser(
        "rm", help="delete its sandbox and snapshot; nothing is kept"
    ).add_argument("id")
    p_wait = sub.add_parser(
        "wait", help="poll until a pushed conversation stops running"
    )
    p_wait.add_argument("id", nargs="?")
    p_wait.add_argument(
        "--every",
        type=float,
        default=WAIT_EVERY,
        help=f"seconds between looks (default {WAIT_EVERY:g})",
    )
    sub.add_parser("setup")
    args = parser.parse_args(argv)

    cwd = Path.cwd().resolve()
    if args.verb == "setup":
        try:
            config = account.setup(
                account.cli_token(interactive=account.interactive()),
                interactive=account.interactive(),
            )
        except account.AccountError as exc:
            print(f"afk: {exc}")
            return 2
        print(
            f"afk: sandboxes now go to {config.project_name}"
            + (f" in {config.team_slug}" if config.team_slug else "")
        )
        return 0
    try:
        gw = gateway()
        # Vercel is needed only once a sandbox is: a push, or a directory
        # with something pushed from it. A first `afk` asks nothing, nor a
        # peek at a conversation here.
        needs_sandbox = args.verb == "push" or bool(
            st.load().for_origin(str(cwd))
        )
        remote_gw = (
            sandbox_gateway(fresh=args.verb == "push")
            if needs_sandbox
            else None
        )
    except (NotAuthenticatedError, account.AccountError) as exc:
        print(f"afk: {exc}")
        return 2
    try:
        # Each verb gathers only what it needs: the list asks everything;
        # push needs this directory's conversations; the rest need afk's
        # own record of what it pushed, and then that one sandbox. Peek and
        # stop work here too: when nothing pushed matches, they look here.
        if args.verb is None:
            if sys.stdin.isatty() and sys.stdout.isatty():
                chosen = await picking(
                    cwd, gw, remote_gw, everything=args.everything
                )
                # Each command only if the one before it succeeded, as
                # `a && b` runs them.
                for command in chosen or []:
                    if command[0] == "afk":
                        code = await main_async(command[1:])
                        if code != 0:
                            return code
                        continue
                    # A harness's own command: afk is done, the terminal is
                    # its.
                    sys.stdout.flush()
                    try:
                        os.execvp(command[0], command)
                    except OSError as exc:
                        print(
                            f"afk: could not run {command[0]}: {exc.strerror}"
                        )
                        return 1
                return 0
            await listing(cwd, gw, remote_gw, everything=args.everything)
            return 0
        if args.verb == "push":
            rows, _, _ = await local_rows(cwd, gw)
            if args.prompt is not None and not args.bg:
                print("afk: --prompt is for --bg; in the TUI, type it there")
                return 2
            row = choose(rows, args.id, where="here", verb="push")
            if row is None:
                return 1
            print(
                f"afk: pushing {row.short} {KIND_SHORT[row.kind]} "
                f"{_title(row)!r}"
            )
            await verbs.push(
                cwd,
                row,
                label=args.label,
                background=args.bg,
                prompt=args.prompt,
                network=args.network,
                git_history=args.git_history,
                hours=_hours(args.hours, gw, remote_gw),
                gateway=remote_gw,
                local_gateway=gw,
            )
            return 0
        if args.verb == "wait":
            return await waiting(cwd, remote_gw, args.id, every=args.every)
        rows = stored_rows(st.load(), cwd)
        if args.verb in ("peek", "stop") and not any(
            r.matches(args.id) for r in rows
        ):
            rows, _, _ = await local_rows(cwd, gw)
        row = choose(rows, args.id, where=None, verb=args.verb)
        if row is None:
            return 1
        if row.where == "here":
            if args.verb == "peek":
                await verbs.peek(row, gw, cwd=cwd)
            else:
                await verbs.stop(row, None)
            return 0
        try:
            if args.verb == "attach":
                await verbs.attach(row, remote_gw)
            elif args.verb == "peek":
                await verbs.peek(row, remote_gw)
            elif args.verb == "pull":
                await verbs.pull(
                    cwd, row, remote_gw, with_files=args.files, local_gateway=gw
                )
            elif args.verb == "stop":
                await verbs.stop(row, remote_gw)
            elif args.verb == "rm":
                await verbs.rm(row, remote_gw)
        except WorkspaceGoneError:
            # These verbs trust afk's record without asking every sandbox
            # first; when the record turns out stale, correct it here.
            state = st.load()
            state.forget_sandbox(row.sandbox or "")
            st.save(state)
            what = (
                "already gone"
                if args.verb in ("stop", "rm")
                else "gone: it no longer exists"
            )
            print(
                f"afk: {row.label or row.short}'s sandbox {row.sandbox} is "
                f"{what}; afk has forgotten it"
            )
            return 0 if args.verb in ("stop", "rm") else 1
        return 0
    except (HarnessError, WorkspaceError) as exc:
        # One line, never a traceback: a sandbox that cannot be reached
        # right now (network, credentials) is a WorkspaceError.
        print(f"afk: {exc}")
        return 1


WAIT_EVERY = 30.0
RUNNING = ("running", "open elsewhere", "?", "")
"""Statuses that mean the agent may still be working."""


async def waiting(
    cwd: Path, remote_gw: Gateway | None, query: str | None, *, every: float
) -> int:
    """Poll until the conversation (or every pushed one that is running)
    stops running: finished, idle, stopped, expired, gone. Each look is
    `info()` per sandbox, which resumes nothing. Ctrl-C stops waiting,
    never the agent. Exit 0 once one has stopped, 1 if none was running."""
    state = st.load()
    targets = stored_rows(state, cwd)
    if query is not None:
        row = choose(targets, query, where="remote", verb="wait")
        if row is None:
            return 1
        targets = [row]
    first, _ = await remote_rows(state, cwd, remote_gw)
    watched = {
        r.session_id
        for r in first
        if r.status in RUNNING
        and any(t.session_id == r.session_id for t in targets)
    }
    if not watched:
        print("afk: nothing pushed from here is running; `afk` lists them")
        return 1
    names = ", ".join(
        r.label or r.short for r in first if r.session_id in watched
    )
    print(f"afk: waiting for {names} · every {every:g}s · Ctrl-C stops waiting")
    while True:
        await asyncio.sleep(every)
        state = st.load()
        now, gone = await remote_rows(state, cwd, remote_gw)
        done = [
            r
            for r in now
            if r.session_id in watched and r.status not in RUNNING
        ]
        for name in gone:
            state.forget_sandbox(name)
        state.changed(now)
        st.save(state)
        if done:
            for r in done:
                cost_ = f" · {r.cost}" if r.cost else ""
                print(f"\a afk: {r.label or r.short} is {r.status}{cost_}")
            return 0


def _hours(
    hours: float, gw: Gateway | None, remote_gw: Gateway | None
) -> float:
    """Cap a push's lifetime at its model credential's, when that is afk's.

    The sandbox injects its credential once, when it is created. A project
    token lives about 12 hours; AI_GATEWAY_API_KEY does not expire.
    """
    if gw is not None or remote_gw is None:
        return hours
    left = (account.seconds_left(remote_gw.credential) - 600) / 3600
    if hours <= left:
        return hours
    print(
        f"afk: the sandbox lives up to {left:.1f}h: that is how long its "
        "Vercel project token lasts (set AI_GATEWAY_API_KEY for longer)"
    )
    return left


def _one_line_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Print an SDK warning in one line, afk's way, without a source trace.

    The SDK warns when something could not travel (e.g. reasoning into
    claude).
    """
    print(f"afk: {message}", file=sys.stderr)


def main() -> None:
    # ty reports identical signatures as incompatible here
    warnings.showwarning = _one_line_warning  # ty: ignore[invalid-assignment]
    try:
        sys.exit(asyncio.run(main_async(sys.argv[1:])))
    except KeyboardInterrupt:
        print()
        sys.exit(130)
