"""The list, as a picker: bare `afk` on a terminal.

Drawn inline, under your prompt like the plain list, never the whole screen.
Choosing returns the commands it stands for — afk verbs, or a harness's own
resume — and the CLI runs them exactly as if you had typed them, with the
terminal back to itself (a harness's TUI needs all of it, and an inline app
cannot hand it over).
Quitting leaves the list in the scrollback; choosing leaves the one-line
command instead.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .rows import KIND_SHORT, Row

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from textual import events

Command = list[str]
Verbs = dict[str, tuple[str, list[Command]]]
"""key -> (what the footer calls it, the commands it runs, in order, each
only if the one before it succeeded: `a && b`)."""

RESUME = {"claude-code": ["claude", "--resume"], "codex": ["codex", "resume"]}
"""Each harness's own command to reopen a conversation in its TUI here."""


def verbs(row: Row) -> Verbs:
    """A key means one thing on every row it is on. Enter puts you in the
    conversation, wherever it is: resumed here, attached there. One open in
    another terminal here, Enter takes over, as attach does there: that
    process ends and the TUI opens here. Where afk cannot end what drives
    it, Enter peeks."""
    sid = row.session_id
    # Peek reads the transcript, so it watches a TUI as well as an
    # unattended agent, here or there.
    peek = ("peek", [["afk", "peek", sid]])
    stop = ("stop", [["afk", "stop", sid]])
    if row.where == "here":
        resume = [*RESUME[row.kind], sid]
        if not row.running:
            enter = ("resume", [resume])
        elif row.pid is not None:
            # Stop waits until the process is gone: never two writers.
            enter = ("take over", [["afk", "stop", sid], resume])
        else:
            enter = peek
        return {
            "enter": enter,
            "p": peek,
            # Stop ends the process that has it open, so only when there is
            # one afk can name.
            **({"s": stop} if row.pid is not None else {}),
            "u": ("push", [["afk", "push", sid]]),
            "b": ("push --bg", [["afk", "push", sid, "--bg"]]),
        }
    return {
        "enter": (
            ("attach", [["afk", "attach", sid]]) if row.mode == "tui" else peek
        ),
        "p": peek,
        "l": ("pull", [["afk", "pull", sid]]),
        "f": ("pull --files", [["afk", "pull", sid, "--files"]]),
        "s": stop,
    }


def line(row: Row) -> Text:
    t = Text(f"{row.short}  {KIND_SHORT[row.kind]:6s} ")
    if row.where == "here":
        t.append(f"{_clip(row.title or '', 44):46s} ")
        t.append(row.status, style="dim")
    else:
        t.append(f"{(row.label or ''):14s} {row.status or '…':14s} ")
        origin = f"← from {row.from_id[:4]} · " if row.from_id else ""
        t.append(f"{origin}{row.sandbox}", style="dim")
    return t


def _clip(s: str, n: int) -> str:
    s = s.strip().replace("\n", " ")
    return s[: n - 1] + "…" if len(s) > n else s


class Picker(App[list[Command] | None]):
    CSS = """
    Screen:inline { border-top: none; border-bottom: none; }
    #head, #keys, OptionList { padding: 0 2; }
    #keys { color: $text-muted; }
    OptionList { border: none; max-height: 20; background: transparent; }
    OptionList:focus { background-tint: transparent; }
    """
    # Ctrl-C leaves, as it does any CLI; Textual's own Ctrl-C copies.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "leave", show=False, priority=True)
    ]

    def __init__(
        self,
        head: Text,
        here: list[Row],
        remote: Awaitable[list[Row]] | None,
    ) -> None:
        super().__init__(ansi_color=True)
        self.head = head
        self.here_rows = here
        self.remote_rows: list[Row] | None = None
        self._remote = remote
        self.confirming: tuple[str, list[Command]] | None = None

    def compose(self) -> ComposeResult:
        yield Static(self.head, id="head")
        yield OptionList(id="rows")
        yield Static(id="keys")

    async def on_mount(self) -> None:
        self.redraw()
        # The sandboxes answer when they answer: what is already known is
        # there to pick from meanwhile.
        if self._remote is not None:
            self.run_worker(self.load(self._remote))

    async def load(self, remote: Awaitable[list[Row]]) -> None:
        self.remote_rows = await remote
        self.redraw()

    def redraw(self) -> None:
        ol = self.query_one(OptionList)
        keep = ol.highlighted_option.id if ol.highlighted_option else None
        options: list[Option] = []
        if self.here_rows:
            options.append(Option(Text("here", style="bold"), disabled=True))
            options += [
                Option(line(r), id=f"here:{r.session_id}")
                for r in self.here_rows
            ]
        if self._remote is not None:
            options.append(Option(Text("remote", style="bold"), disabled=True))
            if self.remote_rows is None:
                options.append(
                    Option(Text("  asking…", style="dim"), disabled=True)
                )
            options += [
                Option(line(r), id=f"remote:{r.session_id}")
                for r in self.remote_rows or []
            ]
        ol.set_options(options)
        ids = [o.id for o in options]
        # Stay on the row you were on; otherwise the first real one.
        target = (
            keep if keep and keep in ids else next((i for i in ids if i), None)
        )
        ol.highlighted = ids.index(target) if target else None
        self.show_keys()

    def selected(self) -> Row | None:
        opt = self.query_one(OptionList).highlighted_option
        if opt is None or opt.id is None:
            return None
        part, sid = opt.id.split(":", 1)
        rows = self.here_rows if part == "here" else self.remote_rows
        return next((r for r in rows or [] if r.session_id == sid), None)

    def show_keys(self) -> None:
        keys = self.query_one("#keys", Static)
        row = self.selected()
        if self.confirming is not None and row is not None:
            name = row.label or row.short
            if self.confirming[0] == "take over":
                keys.update(
                    f"take over {name} from pid {row.pid}? it ends there. y/N"
                )
                return
            what = (
                f"{KIND_SHORT[row.kind]}, pid {row.pid}"
                if row.where == "here"
                else row.sandbox
            )
            keys.update(f"stop {name} ({what})? y/N")
            return
        keys_ = verbs(row) if row else {}
        # A key that does what Enter does works, but is not shown twice.
        shown = [
            f"{k} {name}"
            for k, (name, commands) in keys_.items()
            if k == "enter" or (name, commands) != keys_["enter"]
        ]
        keys.update(" · ".join([*shown, "q quit"]))

    def on_option_list_option_highlighted(self) -> None:
        self.confirming = None
        self.show_keys()

    def on_option_list_option_selected(self) -> None:
        self.act("enter")

    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            return  # the list's own select: on_option_list_option_selected
        if self.confirming is None and event.key in ("q", "escape"):
            await self.action_leave()
            return
        self.act(event.key)

    async def action_leave(self) -> None:
        # The list stays in the scrollback, without the keys.
        await self.query("#keys").remove()
        self.call_after_refresh(self.exit, None)

    def act(self, key: str) -> None:
        # Every key comes through here, Enter too, so a pending question is
        # answered first: only y goes ahead; any other key is the N.
        if self.confirming is not None:
            (_, commands), self.confirming = self.confirming, None
            if key == "y":
                self.done(commands)
            else:
                self.show_keys()
            return
        row = self.selected()
        chosen = verbs(row).get(key) if row else None
        if chosen is None:
            return
        # Anything that ends a process or a sandbox asks first.
        if any(c[:2] == ["afk", "stop"] for c in chosen[1]):
            self.confirming = chosen
            self.show_keys()
        else:
            self.done(chosen[1])

    def done(self, commands: list[Command]) -> None:
        # Leave the commands it stands for, not the list, by the name the
        # list showed: what you would have typed, so the scrollback reads
        # like a shell session. The commands themselves keep the full id:
        # never ambiguous. Only afk knows the list's names; a harness's own
        # command needs the full id.
        row = self.selected()
        name = (row.label or row.short) if row else None

        def typed(argv: Command) -> str:
            if argv[0] != "afk" or row is None:
                return shlex.join(argv)
            return shlex.join(
                [name if name and a == row.session_id else a for a in argv]
            )

        self.exit(
            commands,
            message=Text(
                f"› {' && '.join(typed(c) for c in commands)}",  # noqa: RUF001
                style="dim",
            ),
        )


async def pick(
    head: Text, here: list[Row], remote: Awaitable[list[Row]] | None
) -> list[Command] | None:
    return await Picker(head, here, remote).run_async(
        inline=True, inline_no_clear=True
    )
