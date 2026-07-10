"""A minimal coding agent: the ai SDK + Textual, one bash tool.

The whole app is this file: a bash tool, a transcript that renders
streamed markdown, and an approval prompt.  The user types a message,
the agent streams its reply, and every bash call suspends the run on a
``ToolApproval`` hook until the operator answers y/n — the decision is
resolved in-process with ``ai.resolve_hook()``.

Run it from this directory:

    uv run main.py
"""

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, ClassVar

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

if TYPE_CHECKING:
    # MarkdownStream is textual's incremental renderer for streaming
    # markdown; it has no public import path yet.
    from textual.widgets._markdown import MarkdownStream

MODEL_ID = os.environ.get("MODEL_ID", "anthropic/claude-sonnet-4.6")

SYSTEM_PROMPT = """\
You are a coding assistant running in a terminal TUI.  Your only tool
is bash: use it to explore files, run programs, and apply changes in
the current working directory.  Keep replies short; use code blocks
for code.
"""

MAX_OUTPUT = 50 * 1024  # bash output cap, in bytes
RESULT_PREVIEW_CHARS = 400


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


@ai.tool(require_approval=True)
async def bash(command: str, timeout: float = 120) -> ai.StreamingTextTool:
    """Run a shell command in the current working directory.

    Returns stdout and stderr interleaved, truncated to 50KB.
    Optionally provide a timeout in seconds.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    sent = 0
    note = ""
    while True:
        try:
            chunk = await asyncio.wait_for(
                proc.stdout.read(4096), deadline - loop.time()
            )
        except TimeoutError:
            note = f"[Timed out after {timeout:g}s]"
            break
        if not chunk:
            break
        sent += len(chunk)
        if sent > MAX_OUTPUT:
            note = f"[Truncated at {MAX_OUTPUT // 1024}KB]"
            break
        yield chunk.decode(errors="replace")
    if note:
        proc.kill()
    await proc.wait()
    if not note and proc.returncode:
        note = f"[Exit code: {proc.returncode}]"
    if not note and sent == 0:
        note = "[No output]"
    if note:
        yield ("\n" if sent else "") + note


# ---------------------------------------------------------------------------
# Transcript rendering
# ---------------------------------------------------------------------------


def _short(value: object, limit: int = 200) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_line(part: ai.messages.ToolCallPart) -> str:
    try:
        args: dict[str, Any] = json.loads(part.tool_args or "{}")
    except json.JSONDecodeError:
        args = {"raw": part.tool_args}
    rendered = ", ".join(f"{k}={_short(v, 80)}" for k, v in args.items())
    return f"→ {part.tool_name}({rendered})"


def _result_preview(result: object, *, is_error: bool) -> str:
    text = (
        result if isinstance(result, str) else json.dumps(result, default=str)
    )
    if len(text) > RESULT_PREVIEW_CHARS:
        over = len(text) - RESULT_PREVIEW_CHARS
        text = text[:RESULT_PREVIEW_CHARS] + f"… [+{over} chars]"
    return f"{'✗' if is_error else '←'} {text}"


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_error_text(e) for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


class ToolOutput(textual.widgets.Static):
    """Streamed output of one tool call."""

    def __init__(self) -> None:
        super().__init__("", classes="tool-output")
        self._raw = ""

    def append(self, chunk: str) -> None:
        self._raw += chunk
        self.update(rich.text.Text(self._raw.rstrip("\n")))


class ChatView(textual.containers.VerticalScroll):
    """The transcript: static messages plus one streaming markdown block."""

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        padding: 1 2 0 2;
        scrollbar-size: 1 1;
    }
    ChatView > * {
        margin: 0 0 1 0;
    }
    ChatView > .user {
        background: $surface;
        padding: 0 1;
    }
    ChatView > .thinking {
        color: $text-muted;
        text-style: italic;
    }
    ChatView > .tool {
        color: $text-muted;
    }
    ChatView > .tool-output {
        color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    ChatView > .system {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._stream: MarkdownStream | None = None
        self._stream_role = ""
        self._tool_blocks: dict[str, ToolOutput] = {}

    def push(self, role: str, text: str) -> None:
        """Add a static, plain-text message."""
        self._forget_stream()
        self.mount(textual.widgets.Static(rich.text.Text(text), classes=role))

    async def stream(self, role: str, chunk: str) -> None:
        """Append a chunk to the current markdown block for *role*.

        Opens a new block when the role changes or after any ``push``.
        """
        if self._stream is None or self._stream_role != role:
            await self.stop_stream()
            md = textual.widgets.Markdown(classes=role)
            await self.mount(md)
            self._stream = textual.widgets.Markdown.get_stream(md)
            self._stream_role = role
        await self._stream.write(chunk)

    async def stop_stream(self) -> None:
        """Flush and close the streaming markdown block, if any."""
        stream = self._stream
        self._forget_stream()
        if stream is not None:
            await stream.stop()

    def _forget_stream(self) -> None:
        # Drop the stream reference without flushing — the next
        # ``stream()`` call starts a fresh block below newer messages.
        self._stream = None
        self._stream_role = ""

    def append_tool_output(self, tool_call_id: str, chunk: str) -> None:
        """Append streamed tool output to the block for *tool_call_id*."""
        block = self._tool_blocks.get(tool_call_id)
        if block is None:
            block = ToolOutput()
            self._tool_blocks[tool_call_id] = block
            self.mount(block)
        block.append(chunk)

    def has_tool_output(self, tool_call_id: str) -> bool:
        return tool_call_id in self._tool_blocks


# ---------------------------------------------------------------------------
# Approval prompt
# ---------------------------------------------------------------------------


class ApprovalPrompt(textual.widgets.Static, can_focus=True):
    """y/n prompt for one pending tool-approval hook."""

    DEFAULT_CSS = """
    ApprovalPrompt {
        border: round $warning;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    class Decided(textual.message.Message):
        def __init__(self, hook_id: str, *, granted: bool) -> None:
            super().__init__()
            self.hook_id = hook_id
            self.granted = granted

    def __init__(self, hook: ai.messages.HookPart[Any]) -> None:
        tool = hook.metadata.get("tool", "?")
        kwargs: dict[str, Any] = hook.metadata.get("kwargs") or {}
        body = rich.text.Text()
        body.append(f"approve {tool}? ", style="bold yellow")
        body.append("[y]es  [n]o", style="bold")
        for key, value in kwargs.items():
            body.append(f"\n  {key} = {_short(value)}", style="dim")
        super().__init__(body)
        self.hook_id = hook.hook_id

    def on_key(self, event: textual.events.Key) -> None:
        if event.character in ("y", "n"):
            event.stop()
            self.post_message(
                self.Decided(self.hook_id, granted=event.character == "y")
            )


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


class CodingAgentApp(textual.app.App[None]):
    """Chat TUI wired to a single-tool agent."""

    TITLE = "coding agent"

    CSS = """
    #dock {
        dock: bottom;
        height: auto;
        padding: 0 1 1 1;
    }
    #composer {
        border: round $surface-lighten-2;
    }
    """

    BINDINGS: ClassVar = [
        textual.binding.Binding("escape", "interrupt", "interrupt"),
    ]

    IDLE_PLACEHOLDER = "describe a task…"
    BUSY_PLACEHOLDER = "working… (Enter queues a message, Esc interrupts)"

    def __init__(self) -> None:
        super().__init__()
        self.model = ai.get_model(MODEL_ID)
        self.agent = ai.Agent(tools=[bash])
        self.messages: list[ai.messages.Message] = [
            ai.system_message(SYSTEM_PROMPT)
        ]
        # Messages typed while a turn is streaming; drained one turn
        # at a time so user/assistant alternation stays clean.
        self.pending: list[str] = []
        self._worker: textual.worker.Worker[None] | None = None
        # Approval prompts resolve hooks from the UI task, outside the
        # run's context, so runs use this app-owned registry and the
        # prompt handler passes it to resolve_hook explicitly.
        self._hook_registry = ai.HookRegistry()

    def compose(self) -> textual.app.ComposeResult:
        yield ChatView()
        with textual.containers.Container(id="dock"):
            # Approval prompts get mounted here, above the composer.
            yield textual.widgets.Input(
                placeholder=self.IDLE_PLACEHOLDER, id="composer"
            )

    def on_mount(self) -> None:
        self.chat.push("system", f"model: {MODEL_ID}  cwd: {os.getcwd()}")
        self.chat.anchor()
        self.composer.focus()

    @property
    def chat(self) -> ChatView:
        return self.query_one(ChatView)

    @property
    def composer(self) -> textual.widgets.Input:
        return self.query_one("#composer", textual.widgets.Input)

    # -- input → turns ------------------------------------------------------

    def on_input_submitted(
        self, event: textual.widgets.Input.Submitted
    ) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        self.chat.push("user", text)
        self.pending.append(text)
        if self._worker is None:
            self.run_turns()

    def action_interrupt(self) -> None:
        """Cancel the running turn (Esc)."""
        if self._worker is not None:
            self._worker.cancel()

    @textual.work(exclusive=True, group="turn")
    async def run_turns(self) -> None:
        """Drain queued messages, running one agent turn per message."""
        self._worker = textual.worker.get_current_worker()
        self.composer.placeholder = self.BUSY_PLACEHOLDER
        try:
            while self.pending:
                self.messages.append(ai.user_message(self.pending.pop(0)))
                try:
                    await self._run_turn()
                except asyncio.CancelledError:
                    self.chat.push("system", "interrupted")
                    raise
                except Exception as exc:
                    self.chat.push("system", f"error: {_error_text(exc)}")
        finally:
            self._worker = None
            for prompt in self.query(ApprovalPrompt):
                prompt.remove()
            self.composer.placeholder = self.IDLE_PLACEHOLDER

    async def _run_turn(self) -> None:
        """Run one agent turn, rendering events into the transcript."""
        chat = self.chat
        async with self.agent.run(
            self.model, self.messages, hook_registry=self._hook_registry
        ) as stream:
            try:
                async for event in stream:
                    if isinstance(event, ai.events.ReasoningDelta):
                        await chat.stream("thinking", event.chunk)
                    elif isinstance(event, ai.events.TextDelta):
                        await chat.stream("assistant", event.chunk)
                    elif isinstance(event, ai.events.ToolEnd):
                        chat.push("tool", _tool_line(event.tool_call))
                    elif isinstance(event, ai.events.PartialToolCallResult):
                        if event.tool_call_id is not None:
                            chat.append_tool_output(
                                event.tool_call_id, str(event.value)
                            )
                    elif isinstance(event, ai.events.ToolCallResult):
                        # Show results that didn't stream through
                        # PartialToolCallResult — e.g. denied calls.
                        for part in event.results:
                            if not chat.has_tool_output(part.tool_call_id):
                                chat.push(
                                    "tool-output",
                                    _result_preview(
                                        part.result, is_error=part.is_error
                                    ),
                                )
                    elif isinstance(event, ai.events.HookEvent):
                        self._on_hook(event.hook)
            finally:
                # ``stream.messages`` is always a clean prefix of
                # completed rounds — keep it even on interrupt/error so
                # the next turn resumes from consistent history.
                self.messages = list(stream.messages)
        await chat.stop_stream()

    # -- approval hooks ------------------------------------------------------

    def _on_hook(self, hook: ai.messages.HookPart[Any]) -> None:
        """Mount or dismiss an approval prompt for a hook signal."""
        if hook.status == "pending":
            prompt = ApprovalPrompt(hook)
            self.query_one("#dock").mount(prompt, before=self.composer)
            if not isinstance(self.focused, ApprovalPrompt):
                prompt.focus()
        else:
            self._drop_prompt(hook.hook_id)

    def on_approval_prompt_decided(self, event: ApprovalPrompt.Decided) -> None:
        ai.resolve_hook(
            event.hook_id,
            ai.tools.ToolApproval(
                granted=event.granted,
                reason="approved by operator"
                if event.granted
                else "denied by operator",
            ),
            registry=self._hook_registry,
        )
        self.chat.push("system", "approved" if event.granted else "denied")
        self._drop_prompt(event.hook_id)

    def _drop_prompt(self, hook_id: str) -> None:
        others: list[ApprovalPrompt] = []
        for prompt in self.query(ApprovalPrompt):
            if prompt.hook_id == hook_id:
                prompt.remove()
            else:
                others.append(prompt)
        (others[0] if others else self.composer).focus()


def main() -> None:
    CodingAgentApp().run()


if __name__ == "__main__":
    main()
