"""afk — step away; your agents don't.

    afk                      what's here, what's still running elsewhere
    afk push [id] [--bg]     into a sandbox: the TUI there, or unattended
    afk attach <id>          your terminal, back on its TUI (reopened if it
                             exited)
    afk peek <id>            watch an unattended agent, read-only
    afk pull <id> [--files]  bring it home, in the TUI; --files brings its
                             edits too
    afk stop <id>            end its sandbox
    afk setup                choose the Vercel team afk's sandboxes use

An id is whatever the list showed: a label you gave (`--as`), a word from a
title, or a short id prefix. With none, the one conversation clearly in use
is chosen; with several in play, afk asks. Ctrl-] detaches from any TUI.

On a terminal, bare `afk` is a picker: arrow keys choose a conversation,
Enter or a letter runs the command shown for it, q leaves the list as is.
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
from .rows import KIND_SHORT, Row, local_rows, remote_rows, stored_rows

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
            "  no conversations here yet — start one with `claude` or `codex`"
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
        for r in remote:
            origin = f"← from {r.from_id[:4]} · " if r.from_id else ""
            print(
                f"  {r.short}  {KIND_SHORT[r.kind]:6s} {(r.label or ''):14s} "
                f"{r.status:14s} {DIM}{origin}{r.sandbox}{RESET}"
            )
        if gone:
            for name in gone:
                state.forget_sandbox(name)
            st.save(state)
    print(
        f"\n{DIM}afk push <id> [--bg] · attach <id> · peek <id> · pull <id> "
        f"[--files] · stop <id>{RESET}"
    )


async def picking(
    cwd: Path,
    gw: Gateway | None,
    remote_gw: Gateway | None,
    *,
    everything: bool,
) -> list[str] | None:
    """The list as a picker, on a terminal: the command you chose, to run as
    if typed, or None. The same rows as `listing`, arriving the same way."""
    # textual is imported only when a picker is drawn
    from rich.text import Text  # noqa: PLC0415

    from . import tui  # noqa: PLC0415

    state = st.load()

    async def away() -> list[Row]:
        remote, gone = await remote_rows(state, cwd, remote_gw)
        if gone:
            for name in gone:
                state.forget_sandbox(name)
            st.save(state)
        return remote

    probing = (
        asyncio.create_task(away()) if state.for_origin(str(cwd)) else None
    )
    here, notes, answered = await local_rows(cwd, gw)
    where = f"{BOLD}{_tilde(cwd)}{RESET}  {DIM}·  "
    harnesses = ", ".join(answered) or "no harness found"
    if not here and probing is None:
        print(f"{where}{harnesses}{RESET}")
        for note in notes:
            print(f"  {DIM}{note}{RESET}")
        print(
            "\n  no conversations here yet — start one with `claude` or "
            "`codex`\n"
        )
        return None
    shown = here if everything else here[:LIST_CAP]
    if len(here) > len(shown):
        notes.append(f"… {len(here) - len(shown)} more; `afk --all` shows them")
    head = Text.assemble((_tilde(cwd), "bold"), (f"  ·  {harnesses}", "dim"))
    for note in notes:
        head.append(f"\n  {note}", style="dim")
    return await tui.pick(head, shown, probing)


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
    sub.add_parser("stop").add_argument("id")
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
        # Vercel is needed only once a sandbox is: never for a directory
        # with nothing pushed, so a first `afk` asks nothing.
        needs_sandbox = args.verb is not None or bool(
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
        # own record of what it pushed, and then that one sandbox.
        if args.verb is None:
            if sys.stdin.isatty() and sys.stdout.isatty():
                chosen = await picking(
                    cwd, gw, remote_gw, everything=args.everything
                )
                return 0 if chosen is None else await main_async(chosen)
            await listing(cwd, gw, remote_gw, everything=args.everything)
            return 0
        if args.verb == "push":
            rows, _, _ = await local_rows(cwd, gw)
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
                hours=_hours(args.hours, gw, remote_gw),
                gateway=remote_gw,
                local_gateway=gw,
            )
            return 0
        row = choose(
            stored_rows(st.load(), cwd), args.id, where="remote", verb=args.verb
        )
        if row is None:
            return 1
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
        except WorkspaceGoneError:
            # These verbs trust afk's record without asking every sandbox
            # first; when the record turns out stale, correct it here.
            state = st.load()
            state.forget_sandbox(row.sandbox or "")
            st.save(state)
            what = (
                "already gone"
                if args.verb == "stop"
                else "gone (stopped or expired)"
            )
            print(
                f"afk: {row.label or row.short}'s sandbox {row.sandbox} is "
                f"{what}; afk has forgotten it"
            )
            return 0 if args.verb == "stop" else 1
        return 0
    except (HarnessError, WorkspaceError) as exc:
        # One line, never a traceback: a sandbox that cannot be reached
        # right now (network, credentials) is a WorkspaceError.
        print(f"afk: {exc}")
        return 1


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
