"""tau — a coding-agent chat bot built on the `ai` library and Textual.

Single-process Textual TUI.  The user types a message, it gets appended
to a running conversation history, and the agent streams its reply into
a new assistant bubble.  No tools yet — this is the chat-bot baseline
we'll grow real coding capabilities on top of.

    uv run python tau.py
"""

from __future__ import annotations

import os

import rich.text
import textual
import textual.app
import textual.binding
import textual.containers
import textual.events
import textual.message
import textual.widgets
import textual.worker

import ai

_raw_model = os.environ.get("TAU_MODEL", "gateway:anthropic/claude-opus-4.6")
MODEL_ID = _raw_model if ":" in _raw_model else f"gateway:{_raw_model}"

SYSTEM_PROMPT = """\
You are tau, a focused coding assistant running inside a terminal TUI.
Keep replies concise and use code blocks when showing code.
"""


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
        bubble = app.transcript.add_bubble("assistant")
        try:
            async with app.agent.run(app.model, app.messages) as stream:
                async for event in stream:
                    if isinstance(event, ai.events.TextDelta):
                        # Stay glued to the bottom only if we're already
                        # there — don't yank a scrolled-up reader down.
                        following = app.transcript.at_bottom
                        bubble.append(event.chunk)
                        if following:
                            app.transcript.scroll_end(animate=False)
                # Persist whatever the agent added (assistant + tool turns)
                # so the next turn sees the full history.
                app.messages = list(stream.messages)
        except Exception as exc:  # noqa: BLE001 — surface in the UI
            app.transcript.add_bubble("system", f"error: {exc}")


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

    def __init__(self) -> None:
        super().__init__()
        self.model = ai.get_model(MODEL_ID)
        self.agent = ai.agent()
        # The full conversation, including the system prompt.  We mutate
        # this in place so the agent always sees the entire history.
        self.messages = [ai.system_message(SYSTEM_PROMPT)]
        # User messages typed while a turn is streaming.  Drained one at
        # a time at the end of each turn so user/assistant alternation
        # stays clean.
        self.pending = []
        self._busy = False

    def compose(self) -> textual.app.ComposeResult:
        yield Transcript(id="transcript")
        with textual.containers.Container(id="composer-dock"):
            yield Composer(placeholder="message tau…", id="composer")

    def on_mount(self) -> None:
        self.transcript.add_bubble("system", f"connected — model: {MODEL_ID}")
        self.query_one("#composer", Composer).focus()

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
    # Helpers
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        inp = self.query_one("#composer", Composer)
        # Composer stays enabled while busy — the user can keep typing
        # and queue the next message.  Only the placeholder changes.
        inp.placeholder = (
            "tau is thinking… (type to queue your next message)"
            if busy
            else "message tau…"
        )


if __name__ == "__main__":
    TauApp().run()
