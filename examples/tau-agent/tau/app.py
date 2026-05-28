"""tau — a coding-agent chat bot built on the `ai` library and Textual.

Single-process Textual TUI.  The user types a message, it gets appended
to a running conversation history, and the agent streams its reply into
a new assistant bubble.

Sessions are persisted to ``.tau/sessions/`` as JSONL files and can be
resumed:

    python -m tau                       # new session
    python -m tau --resume              # resume most recent session
    python -m tau --session ID          # resume a specific session
    python -m tau --list                # list saved sessions
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import pathlib
import sys
from typing import Any

import ai
import ai.types.usage
import pydantic
import rich.console
import rich.markdown
import rich.text
import textual
import textual.app
import textual.binding
import textual.containers
import textual.events
import textual.message
import textual.widgets
import textual.worker

from tau import session, tools

_raw_model = os.environ.get("TAU_MODEL", "gateway:anthropic/claude-opus-4.8")
MODEL_ID = _raw_model if ":" in _raw_model else f"gateway:{_raw_model}"


def _provider_slug(model_id: str) -> str:
    """Extract the backend provider slug from a model id.

    "gateway:anthropic/claude-opus-4.8" -> "anthropic",
    "anthropic:claude-..."              -> "anthropic",
    "openai/gpt-..."                    -> "openai".
    """
    mid = model_id.lower()
    if mid.startswith("gateway:"):
        mid = mid[len("gateway:") :]
    return mid.split("/")[0].split(":")[0]


_PROVIDER = _provider_slug(MODEL_ID)

# Providers whose first-party backend is the model's maker.  For these we pin
# the gateway to that provider via ``only`` so it never falls back to another
# backend that also serves the model (e.g. Bedrock/Vertex for Claude).  Other
# models (open-weight ones, say) are served by many backends where the named
# provider isn't the maker, so we leave their routing unrestricted.
#
# The other providers often don't support all the server side
# tools. In practice the main way that this becomes a problem is that
# when there is an error on the main provider, gateway tries falling
# back to other providers which then fail with unhelpful errors about
# the tools not existing.
_PIN_PROVIDERS = frozenset({"anthropic", "openai"})

# Only send gateway-specific options when routing through the gateway.
STREAM_PARAMS: dict[str, Any] | None = (
    {
        "providerOptions": {
            "gateway": {
                "caching": "auto",
                **(
                    {"only": [_PROVIDER]} if _PROVIDER in _PIN_PROVIDERS else {}
                ),
            },
            "anthropic": {
                "thinking": {"type": "enabled", "budget_tokens": 10000}
            },
        }
    }
    if MODEL_ID.startswith("gateway:")
    else None
)


def _provider_tools(model_id: str) -> list[Any]:
    """Return provider-executed tools for the model's backend.

    Anthropic and OpenAI both offer server-side web search.
    The gateway passes through provider-specific tools, so we
    pick the right one based on the underlying provider.
    """
    provider = _provider_slug(model_id)

    if provider == "anthropic":
        from ai.providers.anthropic import tools as ant

        return [ant.web_search()]
    if provider in ("openai", "xai"):
        from ai.providers.openai import tools as oai

        return [oai.web_search()]
    return []


_ADVERTISE = os.environ.get("TAU_ADVERTISE", "") == "1"

_BASE_SYSTEM_PROMPT = (
    """\
You are tau, a focused coding assistant running inside a terminal TUI.
Keep replies concise and use code blocks when showing code.

You have access to the read, write, edit, bash, grep, find, and ls
tools.  Mutating tools (write, edit, bash) require operator approval.
"""
    + (
        "You also have a web_search tool for current information.\n"
        if _provider_tools(MODEL_ID)
        else ""
    )
    + (
        f"""
When writing or suggesting commit messages, always include a trailer line:

    Co-authored-by: {_raw_model}, via tau
"""
        if _ADVERTISE
        else ""
    )
)


def _find_agents_md() -> str | None:
    """Walk up from cwd looking for AGENTS.md; return its contents or None."""
    cur = pathlib.Path.cwd().resolve()
    for directory in (cur, *cur.parents):
        path = directory / "AGENTS.md"
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return None
    return None


def _build_system_prompt() -> str:
    prompt = _BASE_SYSTEM_PROMPT
    agents_md = _find_agents_md()
    if agents_md:
        prompt += (
            "\nThe following project-level instructions were loaded from "
            "AGENTS.md and should be followed:\n\n"
            f"{agents_md}\n"
        )
    return prompt


SYSTEM_PROMPT = _build_system_prompt()

# How many characters of a tool result to show inline; the full result
# still goes to the model.
RESULT_PREVIEW_CHARS = 400


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Owns the message list, session file, and usage bookkeeping.

    Pure data — no UI.  The app reads ``.messages``, ``.session_id``,
    ``.total_usage``, and ``.last_usage`` to drive the display.
    """

    def __init__(self, system_prompt: str) -> None:
        self.messages: list[ai.messages.Message] = [
            ai.system_message(system_prompt)
        ]
        self.session_id: str = ""
        self.total_usage: ai.types.usage.Usage = ai.types.usage.Usage()
        self.last_usage: ai.types.usage.Usage | None = None
        self._session_path: pathlib.Path | None = None
        self._saved_count: int = 0  # messages already written to disk

    # -- lifecycle ---------------------------------------------------------

    def start(self, model_id: str) -> None:
        """Create a new session file and persist the system message."""
        self.session_id = session.new_session_id()
        self._session_path = session.create_session(self.session_id, model_id)
        self._saved_count = session.append_messages(
            self._session_path, self.messages, after=0
        )

    def restore(self, path: pathlib.Path) -> dict[str, Any]:
        """Load an existing session from *path*.

        Returns the metadata dict.  Populates ``.messages`` and
        ``.session_id``; call ``refresh_usage()`` afterwards.
        """
        meta, messages = session.load_messages(path)
        self.session_id = meta.get("session_id", path.stem)
        self._session_path = path
        if messages:
            self.messages = messages
        self._saved_count = len(self.messages)  # already on disk
        return meta

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        """Append any new messages to the session JSONL file."""
        if self._session_path is None:
            return
        self._saved_count = session.append_messages(
            self._session_path, self.messages, after=self._saved_count
        )

    # -- usage -------------------------------------------------------------

    def refresh_usage(self) -> None:
        """Re-derive cumulative usage from all messages."""
        total = ai.types.usage.Usage()
        last: ai.types.usage.Usage | None = None
        for msg in self.messages:
            if msg.usage is not None:
                total = total + msg.usage
                last = msg.usage
        self.total_usage = total
        self.last_usage = last


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def _replay_session(app: TauApp) -> None:
    """Replay persisted messages into the transcript after a restore."""
    for msg in app.session.messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            app.transcript.add_bubble("user", msg.text)
        elif msg.role == "assistant":
            for part in msg.parts:
                if isinstance(part, ai.messages.ReasoningPart):
                    app.transcript.add_bubble("thinking", part.text)
                elif isinstance(part, ai.messages.TextPart):
                    app.transcript.add_bubble("assistant", part.text)
                elif isinstance(part, ai.messages.ToolCallPart):
                    app.transcript.add_bubble(
                        "tool",
                        _format_tool_call(part.tool_name, part.tool_args),
                    )
        elif msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, ai.messages.ToolResultPart):
                    diff = (
                        _format_edit_diff_from_result(part.result)
                        if part.tool_name == "edit"
                        else None
                    )
                    if diff is not None:
                        app.transcript.add_bubble(
                            "tool-result",
                            renderable=diff,
                        )
                    result = part.result
                    if part.tool_name == "edit" and isinstance(result, dict):
                        result = result.get("message", result)
                    app.show_tool_result(result, part.is_error)
    app.show_system(
        f"resumed session {app.session.session_id} "
        f"({len(app.session.messages) - 1} messages) — model: {MODEL_ID}",
    )


async def chat_loop(app: TauApp) -> None:
    """Drain the pending queue, running one agent turn per queued message.

    Reads from ``app.pending`` and ``app.session.messages``; dispatches
    streamed events to app methods for rendering.  All interaction with
    the ``ai`` library lives here.
    """
    while app.pending:
        # Reset bubble state so each turn gets fresh bubbles — otherwise
        # a queued message's response appends into the previous turn's
        # assistant bubble.
        app._reset_turn_bubbles()
        # Pop one queued message into history per turn so the model sees
        # a clean user → assistant → user → … sequence.
        app.session.messages.append(ai.user_message(app.pending.pop(0)))
        app.session.save()
        try:
            await _run_turn(app)
        except asyncio.CancelledError:
            app.show_system("interrupted")
            raise
        except Exception as exc:  # noqa: BLE001 — surface in the UI
            app.show_system(f"error: {_flatten_error(exc)}")


async def _run_turn(app: TauApp) -> None:
    """Execute a single agent turn, dispatching events to the app."""
    interrupted = False
    async with app.agent.run(
        app.model, app.session.messages, params=STREAM_PARAMS
    ) as stream:
        try:
            async for event in stream:
                if isinstance(event, ai.events.ReasoningDelta):
                    app.append_thinking(event.chunk)
                elif isinstance(event, ai.events.TextDelta):
                    app.append_text(event.chunk)
                elif isinstance(event, ai.events.ToolEnd):
                    tc = event.tool_call
                    app.show_tool_call(
                        tc.tool_name,
                        tc.tool_args,
                        event.tool_call_id,
                    )
                elif isinstance(event, ai.events.PartialToolCallResult):
                    if (
                        event.tool_call_id is not None
                        and event.tool_name != "edit"
                    ):
                        app.append_tool_result(
                            event.tool_call_id, str(event.value)
                        )
                elif isinstance(event, ai.events.ToolCallResult):
                    for part in event.results:
                        # Skip if we already streamed this result.
                        if part.tool_call_id not in app._tool_result_bubbles:
                            result = part.result
                            if isinstance(result, tools.EditResult):
                                result = result.message
                            app.show_tool_result(result, part.is_error)
                # -- provider-executed (builtin) tools --
                elif isinstance(event, ai.events.BuiltinToolEnd):
                    btc = event.tool_call
                    app.show_tool_call(btc.tool_name, btc.tool_args)
                elif isinstance(event, ai.events.BuiltinToolResult):
                    app.show_tool_result(
                        event.result.result,
                        event.result.is_error,
                    )
                elif isinstance(event, ai.events.HookEvent):
                    app.on_hook_event(Hook.from_event(event.hook))
        except asyncio.CancelledError:
            interrupted = True
        # Persist whatever the agent added (assistant + tool turns)
        # so the next turn sees the full history.  On interruption we
        # still save the partial state so context isn't lost.
        app.session.messages = list(stream.messages)
        app.session.save()
        app.session.refresh_usage()
        app._update_usage_display()
    if interrupted:
        raise asyncio.CancelledError


def _flatten_error(exc: BaseException) -> str:
    """Unwrap ExceptionGroups and chained exceptions into a readable string."""
    if isinstance(exc, ExceptionGroup):
        parts = [_flatten_error(e) for e in exc.exceptions]
        return "; ".join(parts)
    msg = str(exc)
    if exc.__cause__ is not None:
        msg += f" (caused by {_flatten_error(exc.__cause__)})"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _format_tool_call(name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return f"→ {name}({raw_args})"
    rendered = ", ".join(f"{k}={_short_value(v)}" for k, v in args.items())
    return f"→ {name}({rendered})"


def _short_value(v: Any) -> str:
    if isinstance(v, str):
        s = repr(v)
    else:
        try:
            s = json.dumps(v, ensure_ascii=False)
        except TypeError:
            s = repr(v)
    if len(s) > 80:
        s = s[:77] + "…"
    return s


def _json_default(obj: Any) -> Any:
    """json.dumps fallback: serialize pydantic models via model_dump."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def _format_tool_result(result: Any, is_error: bool) -> str:
    text = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, default=_json_default)
    )
    if len(text) > RESULT_PREVIEW_CHARS:
        text = (
            text[:RESULT_PREVIEW_CHARS]
            + f"… [+{len(text) - RESULT_PREVIEW_CHARS} chars]"
        )
    marker = "✗" if is_error else "←"
    indented = "\n  ".join(text.splitlines() or [""])
    return f"{marker}\n  {indented}"


_DIFF_CTX = 2  # context lines above/below each changed line
_DIFF_ADD = "color(108)"  # green foreground
_DIFF_DEL = "color(168)"  # red/pink foreground


def _render_diff(
    filepath: str,
    old_content: str,
    new_content: str,
) -> rich.text.Text | None:
    """Render a pi-style diff between old and new file content.

    Uses ``difflib.SequenceMatcher`` on lines to collapse unchanged
    regions.  Returns *None* if the contents are identical.
    """
    import difflib

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    gutter_w = len(str(max(len(old_lines), len(new_lines)) + 1)) + 1

    sm = difflib.SequenceMatcher(
        None,
        old_lines,
        new_lines,
        autojunk=False,
    )

    out = rich.text.Text()
    out.append("edit ", style="bold")
    out.append(f"{filepath}\n\n")

    opcodes = sm.get_opcodes()
    for oi, (op, i1, i2, j1, j2) in enumerate(opcodes):
        if op == "equal":
            head_end = i2
            if oi > 0:
                head_end = min(i1 + _DIFF_CTX, i2)
                for k in range(i1, head_end):
                    _diff_ctx(out, gutter_w, k + 1, old_lines[k])
            if oi < len(opcodes) - 1:
                tail_start = max(i2 - _DIFF_CTX, i1)
                tail_start = max(tail_start, head_end)
                if tail_start > head_end:
                    out.append("   ...\n", style="dim")
                for k in range(tail_start, i2):
                    _diff_ctx(out, gutter_w, k + 1, old_lines[k])
        else:
            if op in ("replace", "delete"):
                for k in range(i1, i2):
                    _diff_line(out, gutter_w, "-", k + 1, old_lines[k])
            if op in ("replace", "insert"):
                for k in range(j1, j2):
                    _diff_line(out, gutter_w, "+", k + 1, new_lines[k])

    if out.plain.endswith("\n"):
        out.right_crop(1)
    return out or None


def _format_edit_diff_from_args(
    filepath: str,
    edits: list[dict[str, str]],
) -> rich.text.Text | None:
    """Render an edit diff at ToolEnd time by reading the file from disk."""
    try:
        old_content = pathlib.Path(filepath).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        text_edits = [tools.TextEdit.model_validate(e) for e in edits]
    except pydantic.ValidationError:
        return None
    try:
        new_content = tools.edit_string(old_content, text_edits, filepath)
    except (ValueError, KeyError):
        return None
    return _render_diff(filepath, old_content, new_content)


def _format_edit_diff_from_result(
    result: Any,
) -> rich.text.Text | None:
    """Render an edit diff from a persisted EditResult."""
    if not isinstance(result, dict):
        return None
    try:
        er = tools.EditResult.model_validate(result)
    except (pydantic.ValidationError, Exception):
        return None
    # Infer filepath from the message.
    msg = er.message
    prefix = " in "
    idx = msg.rfind(prefix)
    if idx < 0:
        return None
    filepath = msg[idx + len(prefix) :].rstrip(".")
    return _render_diff(filepath, er.old_content, er.new_content)


def _diff_ctx(
    out: rich.text.Text,
    gw: int,
    ln: int,
    text: str,
) -> None:
    out.append(f"  {ln:>{gw}} {text}\n", style="dim")


def _diff_line(
    out: rich.text.Text,
    gw: int,
    prefix: str,
    ln: int,
    text: str,
) -> None:
    style = _DIFF_ADD if prefix == "+" else _DIFF_DEL
    out.append(f"{prefix} {ln:>{gw}} {text}\n", style=style)


# ---------------------------------------------------------------------------
# Hook — thin wrapper so UI code never touches ai.messages.HookPart
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Hook:
    """UI-facing snapshot of a tool-approval hook."""

    hook_id: str
    tool: str
    kwargs: dict[str, Any]
    status: str

    @classmethod
    def from_event(cls, part: ai.messages.HookPart[Any]) -> Hook:
        return cls(
            hook_id=part.hook_id,
            tool=part.metadata.get("tool", "?"),
            kwargs=part.metadata.get("kwargs", {}) or {},
            status=part.status,
        )


@dataclasses.dataclass
class PromptOption:
    """One option in an approval prompt."""

    key: str  # keyboard shortcut
    decision: str  # value passed to ApprovalTracker.resolve
    label: str  # display text


# ---------------------------------------------------------------------------
# Approval tracking
# ---------------------------------------------------------------------------

# Tools grouped by category for approval purposes.
_READ_TOOLS = frozenset({"read", "grep", "find", "ls"})
_WRITE_TOOLS = frozenset({"write", "edit"})
_FILE_TOOLS = _READ_TOOLS | _WRITE_TOOLS


def _tool_path(hook: Hook) -> pathlib.Path | None:
    """Extract and resolve the path argument from a file-tool hook."""
    raw = hook.kwargs.get("path")
    if raw is None:
        return None
    return pathlib.Path(raw).expanduser().resolve()


class ApprovalTracker:
    """Session-scoped approval state for tool hooks.

    Tracks "always approve" decisions so subsequent identical commands
    (or all commands) can be auto-resolved without prompting.

    File I/O tools are auto-approved when the target path is under the
    working directory.  Paths outside cwd require a prompt; one of the
    options is to permanently allow a directory for reads or writes.
    """

    def __init__(self) -> None:
        self._cwd = pathlib.Path.cwd().resolve()
        self._approve_all = False
        self._approved_commands: set[str] = set()
        # Extra directory trees approved per category.
        self._approved_read_dirs: set[pathlib.Path] = set()
        self._approved_write_dirs: set[pathlib.Path] = set()

    def _path_ok(self, tool: str, path: pathlib.Path | None) -> bool:
        """Check if *path* is in an approved directory for *tool*."""
        if path is None:
            # Tools like grep/find/ls default to cwd when path is None.
            return True
        # Always allow anything under cwd.
        try:
            path.relative_to(self._cwd)
            return True
        except ValueError:
            pass
        # Check extra approved dirs.
        dirs = (
            self._approved_read_dirs
            if tool in _READ_TOOLS
            else self._approved_write_dirs
        )
        return any(path == d or d in path.parents for d in dirs)

    def check(self, hook: Hook) -> bool | list[PromptOption]:
        """Auto-resolve or return prompt options.

        Returns ``True``/``False`` to auto-approve/deny, or a list of
        :class:`PromptOption` when the operator needs to decide.
        """
        if self._approve_all:
            return True
        if hook.tool in _FILE_TOOLS:
            if self._path_ok(hook.tool, _tool_path(hook)):
                return True
            return [
                PromptOption("y", "yes", "yes"),
                PromptOption("n", "no", "no"),
                PromptOption("d", "allow_dir", "allow dir"),
                PromptOption("a", "always_all", "always all"),
            ]
        if hook.tool == "bash":
            cmd = hook.kwargs.get("command", "")
            if cmd in self._approved_commands:
                return True
        return [
            PromptOption("y", "yes", "yes"),
            PromptOption("n", "no", "no"),
            PromptOption("!", "always_this", "always this"),
            PromptOption("a", "always_all", "always all"),
        ]

    def resolve(self, hook: Hook, decision: str) -> bool:
        """Resolve a hook: update approval state, signal the library.

        *decision* is one of ``'yes'``, ``'no'``, ``'always_this'``,
        ``'allow_dir'``, ``'always_all'``.  Returns whether the hook
        was granted.
        """
        # Remember lasting decisions.
        if decision == "always_this":
            cmd = hook.kwargs.get("command", "")
            if cmd:
                self._approved_commands.add(cmd)
        elif decision == "allow_dir":
            path = _tool_path(hook)
            if path is not None:
                directory = path if path.is_dir() else path.parent
                if hook.tool in _READ_TOOLS:
                    self._approved_read_dirs.add(directory)
                elif hook.tool in _WRITE_TOOLS:
                    self._approved_write_dirs.add(directory)
        elif decision == "always_all":
            self._approve_all = True

        granted = decision != "no"
        ai.resolve_hook(
            hook.hook_id,
            ai.tools.ToolApproval(
                granted=granted,
                reason="operator approved" if granted else "operator denied",
            ),
        )
        return granted


def _bell() -> None:
    """Ring the terminal bell to notify the operator."""
    try:
        with open("/dev/tty", "w") as tty:
            tty.write("\a")
            tty.flush()
    except OSError:
        pass


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
    Bubble.tool-result {
        color: $text-muted;
        background: #262626;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    Bubble.thinking {
        color: $text-muted;
        text-style: dim italic;
    }
    """

    def __init__(
        self,
        role: str,
        initial: str = "",
        *,
        renderable: rich.console.RenderableType | None = None,
    ) -> None:
        super().__init__()
        self.add_class(role)
        self._role = role
        self._raw = ""
        self._renderable = renderable
        if renderable is not None:
            self.update(renderable)
        elif initial:
            self.append(initial)
        else:
            self._redraw()

    def append(self, chunk: str) -> None:
        self._raw += chunk
        self._redraw()

    def _redraw(self) -> None:
        if self._role == "assistant":
            self.update(rich.markdown.Markdown(self._raw))
        else:
            self.update(rich.text.Text(self._raw))


class Transcript(textual.containers.VerticalScroll):
    """Scrolling list of bubbles."""

    DEFAULT_CSS = """
    Transcript {
        height: 1fr;
        padding: 1 2 0 2;
        scrollbar-size: 0 0;
    }
    """

    def add_bubble(
        self,
        role: str,
        text: str = "",
        *,
        renderable: rich.console.RenderableType | None = None,
    ) -> Bubble:
        bubble = Bubble(role, text, renderable=renderable)
        self.mount(bubble)
        return bubble


class Composer(textual.widgets.TextArea):
    """Multi-line input that grows with its content.

    Enter submits.  Newline shortcuts:

    - Ctrl+J (all terminals)
    - Trailing backslash before Enter (all terminals)
    - Shift/Alt+Enter (Kitty keyboard protocol)

    Height tracks the wrapped line count between
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

    def _insert_newline(self) -> None:
        """Insert a newline and grow the composer."""
        self.insert("\n")
        self.refresh_height()

    async def _on_key(self, event: textual.events.Key) -> None:
        # Newline shortcuts:
        #  - Ctrl+J (LF — works on all terminals)
        #  - Shift/Alt+Enter (Kitty keyboard protocol)
        if event.key in (
            "ctrl+j",
            "shift+enter",
            "alt+enter",
        ):
            event.stop()
            event.prevent_default()
            self._insert_newline()
            return
        # Plain Enter submits — unless the line ends with \
        # (backslash continuation), in which case strip it and
        # insert a newline instead.
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            if self.text.endswith("\\"):
                # Delete the backslash (cursor is at end)
                # then insert a newline in its place.
                self.action_delete_left()
                self._insert_newline()
                return
            value = self.text
            self.text = ""
            self.refresh_height()
            self.post_message(self.Submitted(value))

    def refresh_height(self) -> None:
        # ``wrapped_document.height`` is the visual line count after soft
        # wrapping.  Clamp it so the composer never collapses to 0 lines
        # or eats the whole screen.  +2 accounts for the top+bottom of
        # the rounded border (box-sizing is border-box by default).
        n = max(
            self.MIN_LINES, min(self.MAX_LINES, self.wrapped_document.height)
        )
        self.styles.height = n + 2


class HookPrompt(textual.widgets.Static):
    """Approval prompt for a pending tool-approval hook.

    Mounts above the composer when a hook fires.  Focusable; single-key
    shortcuts resolve it.  The available options depend on the tool:

    **bash**: ``[y]`` yes ``[n]`` no ``[!]`` always this ``[a]`` all
    **file I/O**: ``[y]`` yes ``[n]`` no ``[d]`` allow dir ``[a]`` all

    Tab/shift-tab cycles focus back to the composer if the user wants
    to look something up before deciding — the hook stays pending and
    the agent stays blocked.
    """

    DEFAULT_CSS = """
    HookPrompt {
        height: auto;
        padding: 0 1;
        border: round $warning;
        margin-bottom: 1;
    }
    HookPrompt:focus {
        border: round $warning;
    }
    """

    can_focus = True

    class Decided(textual.message.Message):
        def __init__(self, hook_id: str, decision: str) -> None:
            super().__init__()
            self.hook_id = hook_id
            self.decision = decision

    def __init__(self, hook: Hook, options: list[PromptOption]) -> None:
        super().__init__()
        self._hook_id = hook.hook_id
        self._options = {opt.key: opt.decision for opt in options}

        body = rich.text.Text()
        body.append("approve ", style="bold yellow")
        body.append(hook.tool, style="bold")
        body.append("?\n")
        body.append("  " + _format_kwargs(hook.kwargs), style="dim")
        body.append("\n  ")
        for i, opt in enumerate(options):
            style = (
                "bold green"
                if opt.decision == "yes"
                else "bold red"
                if opt.decision == "no"
                else "bold cyan"
            )
            if i > 0:
                body.append("  ")
            body.append(f"[{opt.key}]", style=style)
            body.append(f" {opt.label}")
        self.update(body)

    async def _on_key(self, event: textual.events.Key) -> None:
        decision = self._options.get(event.character or "")
        if decision is not None:
            event.stop()
            event.prevent_default()
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
        height: 3;                 /* refresh_height() resizes */
        max-height: 12;            /* MAX_LINES (10) + 2 for the border */
        padding: 0 1;              /* breathing room left/right of the cursor */
        border: round $surface-lighten-2;
    }
    #usage-bar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        textual.binding.Binding("ctrl+c", "quit", "quit", priority=True),
        textual.binding.Binding(
            "escape", "interrupt", "interrupt", priority=True
        ),
    ]

    TITLE = "tau"
    theme = "ansi-dark"

    # State read by ``chat_loop``.  Public on purpose — the agent
    # function is meant to be readable next to the app.
    model: ai.Model
    agent: ai.Agent
    session: SessionManager
    pending: list[str]

    def __init__(
        self,
        *,
        resume_path: pathlib.Path | None = None,
    ) -> None:
        super().__init__()
        self.model = ai.get_model(MODEL_ID)
        self.agent = ai.agent(
            tools=[*tools.TOOLS, *_provider_tools(MODEL_ID)],
        )
        self.session = SessionManager(SYSTEM_PROMPT)
        # User messages typed while a turn is streaming.  Drained one at
        # a time at the end of each turn so user/assistant alternation
        # stays clean.
        self.pending: list[str] = []
        self._busy = False
        # Approval hooks waiting for operator y/n.  FIFO queue: only the
        # head hook is "active" — ``_active_hook`` mirrors it for fast
        # access from the composer.
        self._hook_queue: list[tuple[Hook, list[PromptOption]]] = []
        self._active_hook: Hook | None = None
        self._approval = ApprovalTracker()
        self._turn_worker: textual.worker.Worker[None] | None = None
        self._resume_path = resume_path

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
        self.transcript.anchor()
        self.query_one("#composer", Composer).focus()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _start_new_session(self) -> None:
        self.session.start(MODEL_ID)
        self.show_system(
            f"connected — model: {MODEL_ID}  session: {self.session.session_id}"
        )

    def _restore_session(self, path: pathlib.Path) -> None:
        self.session.restore(path)
        # Replace the persisted system message with the current
        # one so it reflects the latest tools and AGENTS.md.
        if self.session.messages and self.session.messages[0].role == "system":
            self.session.messages[0] = ai.system_message(SYSTEM_PROMPT)
        _replay_session(self)
        self.session.refresh_usage()
        self._update_usage_display()

    def _update_usage_display(self) -> None:
        """Show cumulative token usage in the footer bar."""
        u = self.session.total_usage
        if u.total_tokens == 0:
            return
        parts: list[str] = []
        # Approximate current context size: last turn's in + out.
        if self.session.last_usage is not None:
            ctx = (
                self.session.last_usage.input_tokens
                + self.session.last_usage.output_tokens
            )
            parts.append(f"ctx: ~{ctx:,}")
        # input_tokens includes cache-read; subtract to show uncached.
        uncached_in = u.input_tokens - (u.cache_read_tokens or 0)
        parts.append(f"in: {uncached_in:,}")
        if u.cache_read_tokens:
            parts.append(f"cached: {u.cache_read_tokens:,}")
        parts.append(f"out: {u.output_tokens:,}")
        self.query_one("#usage-bar", textual.widgets.Static).update(
            "  ".join(parts)
        )

    @property
    def transcript(self) -> Transcript:
        return self.query_one("#transcript", Transcript)

    # ------------------------------------------------------------------
    # Rendering — called by the agent loop
    # ------------------------------------------------------------------

    # Per-turn bubble state.  Reset at the start of each turn via
    # ``run_turn``; the agent loop calls the methods below which
    # lazily create bubbles as needed.
    _text_bubble: Bubble | None = None
    _thinking_bubble: Bubble | None = None
    _tool_result_bubbles: dict[str, Bubble] = {}

    def _reset_turn_bubbles(self) -> None:
        self._text_bubble = None
        self._thinking_bubble = None
        self._tool_result_bubbles = {}

    def append_thinking(self, chunk: str) -> None:
        """Append a reasoning/thinking chunk (lazily creates the bubble)."""
        if self._thinking_bubble is None:
            self._thinking_bubble = self.transcript.add_bubble("thinking")
        self._thinking_bubble.append(chunk)

    def append_text(self, chunk: str) -> None:
        """Append an assistant text chunk (lazily creates the bubble)."""
        if self._text_bubble is None:
            self._text_bubble = self.transcript.add_bubble("assistant")
        self._text_bubble.append(chunk)

    def show_tool_call(
        self, name: str, args: str, tool_call_id: str = ""
    ) -> None:
        """Show a completed tool invocation line."""
        rendered = False
        if name == "edit":
            rendered = self._show_edit_diff(args)
        if not rendered:
            self.transcript.add_bubble("tool", _format_tool_call(name, args))
        # Next text from the model should start a fresh bubble so
        # tool output and prose stay visually separated.
        self._text_bubble = None

    def _show_edit_diff(self, args: str) -> bool:
        """Try to render an edit call as a diff.  Returns True on success."""
        try:
            parsed = json.loads(args) if args else {}
        except (json.JSONDecodeError, AttributeError):
            return False
        filepath = parsed.get("path", "")
        edits = parsed.get("edits", [])
        if not filepath or not edits:
            return False
        diff = _format_edit_diff_from_args(filepath, edits)
        if diff is None:
            return False
        self.transcript.add_bubble("tool-result", renderable=diff)
        return True

    def append_tool_result(self, tool_call_id: str, chunk: str) -> None:
        """Append a streaming chunk to a tool-result bubble."""
        bubble = self._tool_result_bubbles.get(tool_call_id)
        if bubble is None:
            bubble = self.transcript.add_bubble("tool-result")
            self._tool_result_bubbles[tool_call_id] = bubble
        bubble.append(chunk)

    def show_tool_result(self, result: Any, is_error: bool) -> None:
        """Show the (possibly truncated) result of a tool call."""
        self.transcript.add_bubble(
            "tool-result", _format_tool_result(result, is_error)
        )

    def show_system(self, text: str) -> None:
        """Show a system/status message."""
        self.transcript.add_bubble("system", text)

    # ------------------------------------------------------------------
    # Input → turn
    # ------------------------------------------------------------------

    def on_text_area_changed(
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
        self._turn_worker = textual.worker.get_current_worker()
        self._reset_turn_bubbles()
        try:
            await chat_loop(self)
        finally:
            self._turn_worker = None
            self._set_busy(False)

    def action_interrupt(self) -> None:
        """Cancel the running turn on ESC."""
        if self._turn_worker is not None:
            self._turn_worker.cancel()
            # Dismiss any pending approval prompt and clear the queue.
            self._hook_queue.clear()
            self._dismiss_active_prompt()

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------

    def _resolve_hook(self, hook: Hook, decision: str) -> None:
        """Resolve a hook and show a transcript note."""
        granted = self._approval.resolve(hook, decision)
        self.show_system(f"{'approved' if granted else 'denied'}: {hook.tool}")

    def on_hook_event(self, hook: Hook) -> None:
        if hook.status == "pending":
            result = self._approval.check(hook)
            if isinstance(result, bool):
                self._resolve_hook(hook, "yes" if result else "no")
                return
            self._hook_queue.append((hook, result))
            self._activate_next_hook()
        elif hook.status in ("resolved", "cancelled"):
            # Drop from queue if it was sitting there waiting.
            self._hook_queue = [
                (h, opts)
                for h, opts in self._hook_queue
                if h.hook_id != hook.hook_id
            ]
            if self._active_hook and self._active_hook.hook_id == hook.hook_id:
                self._dismiss_active_prompt()
                self._activate_next_hook()

    def _activate_next_hook(self) -> None:
        if self._active_hook is not None or not self._hook_queue:
            return
        hook, options = self._hook_queue.pop(0)
        self._active_hook = hook
        prompt = HookPrompt(hook, options)
        dock = self.query_one("#composer-dock", textual.containers.Container)
        composer = self.query_one("#composer", Composer)
        dock.mount(prompt, before=composer)
        prompt.focus()
        _bell()

    def _dismiss_active_prompt(self) -> None:
        for prompt in self.query(HookPrompt).results():
            prompt.remove()
        self._active_hook = None
        self.query_one("#composer", Composer).focus()

    async def on_hook_prompt_decided(self, event: HookPrompt.Decided) -> None:
        hook = self._active_hook
        if hook is None or hook.hook_id != event.hook_id:
            return
        self._resolve_hook(hook, event.decision)
        self._dismiss_active_prompt()
        self._activate_next_hook()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if not busy:
            _bell()
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
    sessions = session.list_sessions()
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


def main() -> None:
    args = _parse_args()

    if args.list:
        _print_sessions()
        sys.exit(0)

    resume_path: pathlib.Path | None = None

    if args.resume:
        resume_path = session.resolve_session(None)
        if resume_path is None:
            print("No sessions to resume.", file=sys.stderr)
            sys.exit(1)
    elif args.session:
        resume_path = session.resolve_session(args.session)
        if resume_path is None:
            print(f"Session not found: {args.session}", file=sys.stderr)
            sys.exit(1)

    TauApp(resume_path=resume_path).run()


if __name__ == "__main__":
    main()
