"""afk — step away; your agents don't.

    afk                      what's here, what's still running elsewhere
    afk push [id] [--bg]     into a sandbox: the TUI there, or unattended
    afk attach <id>          your terminal, back on its TUI (reopened if it
                             exited)
    afk peek <id>            watch an unattended agent, read-only
    afk pull <id> [--files]  bring it home, in the TUI; --files brings its
                             edits too
    afk stop <id>            end its sandbox

An id is whatever the list showed: a label you gave (`--as`), a word from a
title, or a short id prefix. With none, the one conversation clearly in use
is chosen; with several in play, afk asks. Ctrl-] detaches from any TUI.
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
    WorkspaceGoneError,
)

from . import state as st
from . import verbs
from .resolve import AmbiguousError, NoMatchError, match, pick
from .rows import KIND_SHORT, Row, local_rows, remote_rows, stored_rows

if TYPE_CHECKING:
    from typing import TextIO

DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"


def gateway() -> Gateway | None:
    return vercel_ai_gateway() if os.environ.get("AI_GATEWAY_API_KEY") else None


LIST_CAP = 15
DEFAULT_HOURS = 5.0
"""A pushed sandbox's lifetime.

The platform's own default is five minutes, which is shorter than any time away;
five hours is its maximum on Pro.
"""


async def listing(cwd: Path, gw: Gateway | None, *, everything: bool) -> None:
    """The list, drawn as it arrives: this directory's conversations as soon
    as the local stores answer, the pushed ones once their sandboxes do —
    a slow sandbox never holds up what is already known."""
    state = st.load()
    away = (
        asyncio.create_task(remote_rows(state, cwd, gw))
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
    args = parser.parse_args(argv)

    cwd = Path.cwd().resolve()
    try:
        gw = gateway()
    except NotAuthenticatedError as exc:
        print(f"afk: {exc}")
        return 2
    try:
        # Each verb gathers only what it needs: the list asks everything;
        # push needs this directory's conversations; the rest need afk's
        # own record of what it pushed, and then that one sandbox.
        if args.verb is None:
            await listing(cwd, gw, everything=args.everything)
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
                hours=args.hours,
                gateway=gw,
            )
            return 0
        row = choose(
            stored_rows(st.load(), cwd), args.id, where="remote", verb=args.verb
        )
        if row is None:
            return 1
        try:
            if args.verb == "attach":
                await verbs.attach(row, gw)
            elif args.verb == "peek":
                await verbs.peek(row, gw)
            elif args.verb == "pull":
                await verbs.pull(cwd, row, gw, with_files=args.files)
            elif args.verb == "stop":
                await verbs.stop(row, gw)
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
    except HarnessError as exc:
        print(f"afk: {exc}")
        return 1


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
