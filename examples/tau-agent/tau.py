"""tau — a coding-agent chat bot built on the `ai` library and Textual.

Single-process Textual TUI.  The user types a message, it gets appended
to a running conversation history, and the agent streams its reply into
a new assistant bubble.

Sessions are persisted to ``.tau/sessions/`` as JSONL files and can be
resumed:

    uv run python tau.py                # new session
    uv run python tau.py --resume       # resume most recent session
    uv run python tau.py --session ID   # resume a specific session
    uv run python tau.py --list         # list saved sessions
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import ai
import ai.types.usage
import rich.text
import textual
import textual.app
import textual.binding
import textual.containers
import textual.events
import textual.message
import textual.widgets
import textual.worker

import session as session_
import tools as tools_

MODEL_ID = os.environ.get("TAU_MODEL", "anthropic/claude-sonnet-4.6")

STREAM_PARAMS: dict[str, Any] = {
    "providerOptions": {"gateway": {"caching": "auto"}},
}

_ADVERTISE = os.environ.get("TAU_ADVERTISE", "") == "1"

SYSTEM_PROMPT = """\
You are tau, a focused coding assistant running inside a terminal TUI.
Keep replies concise and use code blocks when showing code.

You have access to the read, write, edit, bash, grep, find, and ls
tools.  Mutating tools (write, edit, bash) require operator approval.
""" + (
    f"""
When writing or suggesting commit messages, always include a trailer line:

    Co-authored-by: {MODEL_ID}, via tau
"""
    if _ADVERTISE
    else ""
)

# How many characters of a tool result to show inline; the full result
# still goes to the model.
RESULT_PREVIEW_CHARS = 400


# ===========================================================================
# Agent loop — the only place that touches the `ai` library.
#
# Everything below this function is plain Textual widgets and app plumbing.
# Read this function to understand what tau does; read the rest to
# understand how the TUI renders it.
# ===========================================================================


async def chat_loop(app: TauApp) -> None:
    """Drain the pending queue, running one agent turn per queued message.

    Reads from ``app.pending`` and ``app.messages``; writes streamed
    text into a fresh assistant bubble on ``app.transcript``.  All
    interaction with the ``ai`` library lives here.
    """
    while app.pending:
        # Pop one queued message into history per turn so the model sees
        # a clean user → assistant → user → … sequence.
        app.messages.append(ai.user_message(app.pending.pop(0)))
        app._save_messages()
        try:
            await _run_turn(app)
        except Exception as exc:  # noqa: BLE001 — surface in the UI
            app.transcript.add_bubble("system", f"error: {exc}")


async def _run_turn(app: TauApp) -> None:
    """Execute a single agent turn, streaming events into the transcript."""
    # One assistant bubble per turn for streamed text; tool calls
    # get their own bubbles below.
    text_bubble: Bubble | None = None
    tool_bubbles: dict[str, Bubble] = {}
    async with app.agent.run(app.model, app.messages, params=STREAM_PARAMS) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                if text_bubble is None:
                    text_bubble = app.transcript.add_bubble("assistant")
                following = app.transcript.at_bottom
                text_bubble.append(event.chunk)
                if following:
                    app.transcript.scroll_end(animate=False)
            elif isinstance(event, ai.events.ToolEnd):
                tc = event.tool_call
                bubble = app.transcript.add_bubble(
                    "tool", _format_tool_call(tc.tool_name, tc.tool_args)
                )
                tool_bubbles[tc.tool_call_id] = bubble
                # The next text chunk should start a fresh bubble
                # so tool output and prose stay separated.
                text_bubble = None
            elif isinstance(event, ai.events.ToolCallResult):
                for part in event.results:
                    tb: Bubble | None = tool_bubbles.get(part.tool_call_id)
                    if tb is None:
                        tb = app.transcript.add_bubble(
                            "tool",
                            f"→ {part.tool_name}(?)",
                        )
                    tb.append(_format_tool_result(part.result, part.is_error))
            elif isinstance(event, ai.events.HookEvent):
                app.on_hook_event(event.hook)
        # Persist whatever the agent added (assistant + tool turns)
        # so the next turn sees the full history.
        app.messages = list(stream.messages)
        app._save_messages()
        # Accumulate token usage from new messages.
        app._refresh_usage()


def _format_tool_call(name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return f"→ {name}({raw_args})"
    rendered = ", ".join(f"{k}={_short_value(v)}" for k, v in args.items())
    return f"→ {name}({rendered})"


def _short_value(v: Any) -> str:
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else repr(v)
    if len(s) > 80:
        s = s[:77] + "…"
    return s


def _format_tool_result(result: Any, is_error: bool) -> str:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if len(text) > RESULT_PREVIEW_CHARS:
        text = (
            text[:RESULT_PREVIEW_CHARS]
            + f"… [+{len(text) - RESULT_PREVIEW_CHARS} chars]"
        )
    marker = "✗" if is_error else "←"
    indented = "\n  ".join(text.splitlines() or [""])
    return f"\n  {marker} {indented}"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class Bubble(textual.widgets.Static):
    """One message in the transcript.  Role drives the styling."""

    DEFAULT_CSS = """
    Bubble {
        width: 1fr;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    Bubble.user {
        color: $text;
    }
    Bubble.assistant {
        color: $accent;
    }
    Bubble.system {
        color: $text-muted;
        text-style: italic;
    }
    Bubble.tool {
        color: $text-muted;
    }
    """

    def __init__(self, role: str, initial: str = "") -> None:
        super().__init__()
        self.add_class(role)
        self._role = role
        self._text = rich.text.Text()
        if initial:
            self.append(initial)
        else:
            self._redraw()

    def append(self, chunk: str) -> None:
        self._text.append(chunk)
        self._redraw()

    def _redraw(self) -> None:
        self.update(self._text)


class Transcript(textual.containers.VerticalScroll):
    """Scrolling list of bubbles."""

    DEFAULT_CSS = """
    Transcript {
        height: 1fr;
        padding: 1 2 0 2;
        scrollbar-size: 0 0;
    }
    """

    def add_bubble(self, role: str, text: str = "") -> Bubble:
        bubble = Bubble(role, text)
        self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    @property
    def at_bottom(self) -> bool:
        """True when the scrollback is at (or within 1 row of) the end.

        Used to decide whether streaming text should auto-scroll: if
        the user has scrolled up to read earlier output, we don't yank
        them back down on every chunk.
        """
        return self.scroll_y >= self.max_scroll_y - 1


class Composer(textual.widgets.TextArea):
    """Multi-line input that grows with its content.

    Enter submits.  Shift+Enter (or alt+enter, depending on terminal)
    inserts a newline.  Height tracks the wrapped line count between
    ``MIN_LINES`` and ``MAX_LINES``.
    """

    MIN_LINES = 1
    MAX_LINES = 10

    class Submitted(textual.message.Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, *, placeholder: str = "", id: str | None = None) -> None:
        super().__init__(
            id=id,
            placeholder=placeholder,
            soft_wrap=True,
            show_line_numbers=False,
            # No compact=True: compact mode sets `border: none !important`
            # which would override the rounded border we draw below.
        )

    def on_mount(self) -> None:
        self.refresh_height()

    async def _on_key(self, event: textual.events.Key) -> None:
        # Plain enter submits; shift+enter inserts a newline.
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            value = self.text
            self.text = ""
            self.refresh_height()
            self.post_message(self.Submitted(value))

    def refresh_height(self) -> None:
        # ``wrapped_document.height`` is the visual line count after soft
        # wrapping.  Clamp it so the composer never collapses to 0 lines
        # or eats the whole screen.  +2 accounts for the top+bottom of
        # the rounded border (box-sizing is border-box by default).
        n = max(self.MIN_LINES, min(self.MAX_LINES, self.wrapped_document.height))
        self.styles.height = n + 2


class ApprovalTracker:
    """Session-scoped approval state for tool hooks.

    Tracks "always approve" decisions so subsequent identical commands
    (or all commands) can be auto-resolved without prompting.
    """

    def __init__(self) -> None:
        self._approve_all = False
        self._approved_commands: set[str] = set()

    def check(self, hook: ai.messages.HookPart[Any]) -> bool | None:
        """Return True to auto-approve, False to auto-deny, None to prompt."""
        if self._approve_all:
            return True
        tool = hook.metadata.get("tool", "")
        kwargs = hook.metadata.get("kwargs", {}) or {}
        if tool == "bash":
            cmd = kwargs.get("command", "")
            if cmd in self._approved_commands:
                return True
        return None

    def approve_command(self, hook: ai.messages.HookPart[Any]) -> None:
        """Remember to always approve this exact command."""
        kwargs = hook.metadata.get("kwargs", {}) or {}
        cmd = kwargs.get("command", "")
        if cmd:
            self._approved_commands.add(cmd)

    def approve_all(self) -> None:
        """Auto-approve every future tool call this session."""
        self._approve_all = True


class HookPrompt(textual.widgets.Static):
    """Approval prompt for a pending tool-approval hook.

    Mounts above the composer when a hook fires.  Focusable; single-key
    shortcuts resolve it:

    - ``y`` approve once
    - ``n`` deny
    - ``!`` always approve this exact command
    - ``a`` always approve all commands

    Tab/shift-tab cycles focus back to the composer if the user wants
    to look something up before deciding — the hook stays pending and
    the agent stays blocked.
    """

    DEFAULT_CSS = """
    HookPrompt {
        height: auto;
        padding: 0 1;
        border: round $warning;
        background: $surface;
        margin-bottom: 1;
    }
    HookPrompt:focus {
        border: round $warning;
        background: $surface-lighten-1;
    }
    """

    BINDINGS = [
        textual.binding.Binding("y", "decide('yes')", "approve", show=True),
        textual.binding.Binding("n", "decide('no')", "deny", show=True),
        textual.binding.Binding(
            "exclamation_mark", "decide('always_this')", "always this", show=True
        ),
        textual.binding.Binding("a", "decide('always_all')", "always all", show=True),
    ]

    can_focus = True

    class Decided(textual.message.Message):
        def __init__(self, hook_id: str, decision: str) -> None:
            super().__init__()
            self.hook_id = hook_id
            self.decision = decision  # 'yes' | 'no' | 'always_this' | 'always_all'

    def __init__(self, hook: ai.messages.HookPart[Any]) -> None:
        super().__init__()
        self._hook_id = hook.hook_id
        tool = hook.metadata.get("tool", "?")
        kwargs = hook.metadata.get("kwargs", {}) or {}
        body = rich.text.Text()
        body.append("approve ", style="bold yellow")
        body.append(tool, style="bold")
        body.append("?\n")
        body.append("  " + _format_kwargs(kwargs), style="dim")
        body.append("\n  ")
        body.append("[y]", style="bold green")
        body.append(" yes  ")
        body.append("[n]", style="bold red")
        body.append(" no  ")
        body.append("[!]", style="bold cyan")
        body.append(" always this  ")
        body.append("[a]", style="bold cyan")
        body.append(" always all")
        self.update(body)

    def action_decide(self, decision: str) -> None:
        self.post_message(self.Decided(self._hook_id, decision))


def _format_kwargs(kwargs: dict[str, Any]) -> str:
    return ", ".join(f"{k}={_short_value(v)}" for k, v in kwargs.items()) or "—"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class TauApp(textual.app.App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #composer-dock {
        dock: bottom;
        height: auto;
        layout: vertical;
        /* dock: bottom ignores horizontal margins, so the inset lives here */
        padding: 0 1 1 1;
    }
    #composer {
        height: 3;                 /* refresh_height() resizes this dynamically */
        max-height: 12;            /* MAX_LINES (10) + 2 for the border */
        padding: 0 1;              /* breathing room left/right of the cursor */
        border: round $surface-lighten-2;
        background: $surface;
    }
    #usage-bar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        textual.binding.Binding("ctrl+c", "quit", "quit", priority=True),
        textual.binding.Binding("ctrl+d", "quit", "quit", priority=True),
    ]

    TITLE = "tau"

    # State read by ``chat_loop``.  Public on purpose — the agent
    # function is meant to be readable next to the app.
    model: ai.Model
    agent: ai.Agent
    messages: list[ai.messages.Message]
    pending: list[str]

    def __init__(
        self,
        *,
        resume_path: pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.model = ai.ai_gateway(MODEL_ID)
        self.agent = ai.agent(tools=tools_.TOOLS)
        # The full conversation, including the system prompt.  We mutate
        # this in place so the agent always sees the entire history.
        self.messages: list[ai.messages.Message] = [ai.system_message(SYSTEM_PROMPT)]
        # User messages typed while a turn is streaming.  Drained one at
        # a time at the end of each turn so user/assistant alternation
        # stays clean.
        self.pending: list[str] = []
        self._busy = False
        # Approval hooks waiting for operator y/n.  FIFO queue: only the
        # head hook is "active" — ``_active_hook`` mirrors it for fast
        # access from the composer.
        self._hook_queue: list[ai.messages.HookPart[Any]] = []
        self._active_hook: ai.messages.HookPart[Any] | None = None
        self._approval = ApprovalTracker()

        # Session history — every message is persisted to a JSONL file.
        self._resume_path = resume_path
        self._session_id: str = ""
        self._session_path: pathlib.Path | None = None
        self._saved_count: int = 0  # messages already written to disk
        self._total_usage: ai.types.usage.Usage = ai.types.usage.Usage()
        self._last_usage: ai.types.usage.Usage | None = None

    def compose(self) -> textual.app.ComposeResult:
        yield Transcript(id="transcript")
        with textual.containers.Container(id="composer-dock"):
            # Hook prompts get mounted here via ``before=#composer``.
            yield Composer(placeholder="message tau…", id="composer")
            yield textual.widgets.Static("", id="usage-bar")

    def on_mount(self) -> None:
        if self._resume_path is not None:
            self._restore_session(self._resume_path)
        else:
            self._start_new_session()
        self.query_one("#composer", Composer).focus()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _start_new_session(self) -> None:
        self._session_id = session_.new_session_id()
        self._session_path = session_.create_session(self._session_id, MODEL_ID)
        # Persist the system message.
        self._saved_count = session_.append_messages(
            self._session_path, self.messages, after=0
        )
        self.transcript.add_bubble(
            "system",
            f"connected — model: {MODEL_ID}  session: {self._session_id}",
        )

    def _restore_session(self, path: pathlib.Path) -> None:
        meta, messages = session_.load_messages(path)
        self._session_id = meta.get("session_id", path.stem)
        self._session_path = path
        if messages:
            self.messages = messages
        self._saved_count = len(self.messages)  # already on disk
        # Replay conversation into transcript bubbles.
        for msg in self.messages:
            if msg.role == "system":
                continue  # don't clutter the UI with the system prompt
            if msg.role == "user":
                self.transcript.add_bubble("user", msg.text)
            elif msg.role == "assistant":
                self.transcript.add_bubble("assistant", msg.text)
            elif msg.role == "tool":
                for part in msg.parts:
                    if hasattr(part, "result"):
                        preview = _format_tool_result(
                            part.result, getattr(part, "is_error", False)
                        )
                        self.transcript.add_bubble("tool", preview)
        self.transcript.add_bubble(
            "system",
            f"resumed session {self._session_id} "
            f"({len(self.messages) - 1} messages) — model: {MODEL_ID}",
        )
        self._refresh_usage()

    def _save_messages(self) -> None:
        """Append any new messages to the session JSONL file."""
        if self._session_path is None:
            return
        self._saved_count = session_.append_messages(
            self._session_path, self.messages, after=self._saved_count
        )

    def _refresh_usage(self) -> None:
        """Re-derive cumulative usage from all messages."""
        total = ai.types.usage.Usage()
        last_usage: ai.types.usage.Usage | None = None
        for msg in self.messages:
            if msg.usage is not None:
                total = total + msg.usage
                last_usage = msg.usage
        self._total_usage = total
        self._last_usage = last_usage
        self._update_usage_display()

    def _update_usage_display(self) -> None:
        """Show cumulative token usage in the footer bar."""
        u = self._total_usage
        if u.total_tokens == 0:
            return
        parts: list[str] = []
        # Approximate current context size: last turn's in + out.
        if self._last_usage is not None:
            ctx = self._last_usage.input_tokens + self._last_usage.output_tokens
            parts.append(f"ctx: ~{ctx:,}")
        # input_tokens includes cache-read; subtract to show uncached.
        uncached_in = u.input_tokens - (u.cache_read_tokens or 0)
        parts.append(f"in: {uncached_in:,}")
        if u.cache_read_tokens:
            parts.append(f"cached: {u.cache_read_tokens:,}")
        parts.append(f"out: {u.output_tokens:,}")
        self.query_one("#usage-bar", textual.widgets.Static).update("  ".join(parts))

    @property
    def transcript(self) -> Transcript:
        return self.query_one("#transcript", Transcript)

    # ------------------------------------------------------------------
    # Input → turn
    # ------------------------------------------------------------------

    async def on_text_area_changed(
        self, event: textual.widgets.TextArea.Changed
    ) -> None:
        # Grow/shrink the composer as the user types or wraps.
        if isinstance(event.text_area, Composer):
            event.text_area.refresh_height()

    async def on_composer_submitted(self, event: Composer.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        self.transcript.add_bubble("user", text)
        # All submissions enter the queue; ``run_turn`` is the sole
        # consumer.  The user bubble shows up immediately so the message
        # feels sent even when it won't reach the model until the
        # current turn finishes.
        self.pending.append(text)

        if not self._busy:
            self._set_busy(True)
            self.run_turn()

    @textual.work(exclusive=True, group="turn")
    async def run_turn(self) -> None:
        try:
            await chat_loop(self)
        finally:
            self._set_busy(False)

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------

    def on_hook_event(self, hook: ai.messages.HookPart[Any]) -> None:
        if hook.status == "pending":
            # Check if the tracker can auto-resolve this hook.
            decision = self._approval.check(hook)
            if decision is not None:
                self._resolve_hook(hook, granted=decision)
                return
            self._hook_queue.append(hook)
            self._activate_next_hook()
        elif hook.status in ("resolved", "cancelled"):
            # Drop from queue if it was sitting there waiting.
            self._hook_queue = [
                h for h in self._hook_queue if h.hook_id != hook.hook_id
            ]
            if self._active_hook and self._active_hook.hook_id == hook.hook_id:
                self._dismiss_active_prompt()
                self._activate_next_hook()

    def _activate_next_hook(self) -> None:
        if self._active_hook is not None or not self._hook_queue:
            return
        hook = self._hook_queue.pop(0)
        self._active_hook = hook
        prompt = HookPrompt(hook)
        dock = self.query_one("#composer-dock", textual.containers.Container)
        composer = self.query_one("#composer", Composer)
        dock.mount(prompt, before=composer)
        prompt.focus()

    def _dismiss_active_prompt(self) -> None:
        for prompt in self.query(HookPrompt).results():
            prompt.remove()
        self._active_hook = None
        self.query_one("#composer", Composer).focus()

    def _resolve_hook(self, hook: ai.messages.HookPart[Any], *, granted: bool) -> None:
        """Resolve a hook and show a transcript note."""
        ai.resolve_hook(
            hook.hook_id,
            ai.tools.ToolApproval(
                granted=granted,
                reason="operator approved" if granted else "operator denied",
            ),
        )
        self.transcript.add_bubble(
            "system",
            f"{'approved' if granted else 'denied'}: {hook.metadata.get('tool', '?')}",
        )

    async def on_hook_prompt_decided(self, event: HookPrompt.Decided) -> None:
        hook = self._active_hook
        if hook is None or hook.hook_id != event.hook_id:
            return
        granted = event.decision != "no"
        if event.decision == "always_this":
            self._approval.approve_command(hook)
        elif event.decision == "always_all":
            self._approval.approve_all()
        self._resolve_hook(hook, granted=granted)
        self._dismiss_active_prompt()
        self._activate_next_hook()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        # Composer stays enabled while busy — the user can keep typing
        # and queue the next message.  Only the placeholder changes.
        inp = self.query_one("#composer", Composer)
        inp.placeholder = (
            "tau is thinking… (type to queue your next message)"
            if busy
            else "message tau…"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tau",
        description="tau — a coding-agent TUI",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        "-r",
        action="store_true",
        default=False,
        help="Resume the most recent session.",
    )
    group.add_argument(
        "--session",
        "-s",
        metavar="ID",
        default=None,
        help="Resume a specific session by ID (or unique prefix).",
    )
    group.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        help="List saved sessions and exit.",
    )
    return parser.parse_args()


def _print_sessions() -> None:
    sessions = session_.list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    print(f"{'SESSION ID':<20} {'MODEL':<35} {'CWD'}")
    print("─" * 80)
    for s in sessions:
        sid = s.get("session_id", "?")
        model = s.get("model", "?")
        cwd = s.get("cwd", "?")
        print(f"{sid:<20} {model:<35} {cwd}")


if __name__ == "__main__":
    args = _parse_args()

    if args.list:
        _print_sessions()
        sys.exit(0)

    resume_path: pathlib.Path | None = None

    if args.resume:
        resume_path = session_.resolve_session(None)
        if resume_path is None:
            print("No sessions to resume.", file=sys.stderr)
            sys.exit(1)
    elif args.session:
        resume_path = session_.resolve_session(args.session)
        if resume_path is None:
            print(f"Session not found: {args.session}", file=sys.stderr)
            sys.exit(1)

    TauApp(resume_path=resume_path).run()
