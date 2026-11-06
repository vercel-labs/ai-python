"""Pi, driven through `pi --mode rpc`.

One pi process per conversation: an RPC process holds one session. The id we
hand pi (`--session-id`) IS its session id, so `pi --session-id <id>` opens
the same conversation in pi's own TUI.

pi has no permission prompt. Approval is an extension loaded into every pi
a worker starts (`pi_extension.js`): its `tool_call` handler asks through an
RPC dialog, and this adapter answers it. The protocol shapes here were
measured against a real pi (see VERIFIED_VERSION).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import json
import os
import pathlib
import re
import uuid
import warnings
from collections.abc import AsyncIterator
from typing import Any

from ...types import events as events_
from ...types import messages as messages_
from ...types import usage as usage_
from .. import _bridge
from .. import _errors as errors
from . import _handoff, base

KIND = "pi"
INSTALL = "npm install -g @earendil-works/pi-coding-agent"
VERIFIED_VERSION = "1.3.0"
"""The CLI the live tests last passed against. Older ones get a warning."""
EXTENSION = str(pathlib.Path(__file__).with_name("pi_extension.js"))
APPROVE = "ai-python:approve"
"""The dialog title the extension asks approvals under."""
READY = "ai-python-ready"
"""The command the extension registers last: proof that it loaded."""
CONFIG_VAR = "AI_PYTHON_PI"
"""The environment variable that configures the extension."""
SESSION_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,198}[A-Za-z0-9])?\Z")
"""pi's rule for a session id: letters, digits, `.`, `_` and `-`, starting
and ending with a letter or digit. Never a path."""
DIALOGS = ("select", "confirm", "input", "editor")
"""Extension UI methods that wait for an answer; the rest are notices."""
CRASHED = "__crashed__"
CLOSED = "__closed__"

AUTH_HINT = (
    "pi uses its own login (`/login`, auth.json) or a provider key in the "
    "environment: pass env={'ANTHROPIC_API_KEY': ...}."
)


class PiAdapter:
    def __init__(
        self,
        *,
        cwd: str,
        env: dict[str, str],
        executable: str,
        model: str | None,
        effort: str | None,
        mcp_servers: dict[str, Any],
        approve: base.Approve | None,
        allowed_tools: list[str] | None,
        disallowed_tools: list[str] | None,
    ) -> None:
        self._cwd = cwd
        self._env = env
        self._executable = executable
        self._model = model
        self._effort = effort
        self._mcp_servers = mcp_servers
        self._approve = approve
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools
        self._instructions: str | None = None
        self._agent: _Agent | None = None
        # We ended the turn: it settles as cancelled.
        self._interrupted = False
        # A prompt was accepted and no `agent_settled` came yet.
        self._running = False
        self.version: str | None = None

    async def start(self) -> None:
        self.version = await base.probe(
            KIND,
            self._executable,
            install=INSTALL,
            verified=VERIFIED_VERSION,
            cwd=self._cwd,
            env=self._env,
        )

    async def close(self) -> None:
        agent, self._agent = self._agent, None
        self._running = False
        if agent is not None:
            await agent.close()
            with contextlib.suppress(Exception):
                await agent.process.terminate()

    def _args(self) -> list[str]:
        """How pi runs: headless and in the TUI alike."""
        args = ["-e", EXTENSION]
        if self._model:
            args += ["--model", self._model]
        if self._effort:
            args += ["--thinking", self._effort]
        if self._instructions:
            args += ["--append-system-prompt", self._instructions]
        if self._allowed_tools is not None:
            args += (
                ["--tools", ",".join(self._allowed_tools)]
                if self._allowed_tools
                else ["--no-tools"]
            )
        if self._disallowed_tools:
            args += ["--exclude-tools", ",".join(self._disallowed_tools)]
        return args

    def _extension_env(
        self, *, approve: bool, servers: dict[str, Any]
    ) -> dict[str, str]:
        config: dict[str, Any] = {"approve": approve}
        if servers:
            config["mcp"] = mcp_config(servers)
        return {**self._env, CONFIG_VAR: json.dumps(config)}

    # -- the conversation -----------------------------------------------------

    async def open(
        self,
        *,
        session_id: str | None,
        fork: bool,
        seed: list[messages_.Message] | None,
        instructions: str | None,
    ) -> str:
        self._instructions = instructions
        if seed:
            session_id = await self._stage(seed)
            args = ["--session-id", session_id]
        elif session_id is not None:
            # Checked first: `--session-id` CREATES a session it cannot
            # find, so a typo would silently start an empty conversation.
            found = await self._find(session_id)
            if fork:
                session_id = str(uuid.uuid4())
                # By path, not id: pi matches a partial id, a path is exact.
                args = ["--fork", found["path"], "--session-id", session_id]
            else:
                args = ["--session-id", session_id]
        else:
            session_id = str(uuid.uuid4())
            args = ["--session-id", session_id]
        await self._spawn(session_id, args)
        return session_id

    async def _spawn(self, session_id: str, args: list[str]) -> None:
        """Start pi on one conversation and wait until it answers.

        A pi that cannot start (an unknown model, a bad flag) exits at once,
        and the handshake reports its last words.
        """
        process = await base.spawn(
            [self._executable, "--mode", "rpc", *args, *self._args()],
            cwd=self._cwd,
            env=self._extension_env(
                approve=self._approve is not None, servers=self._mcp_servers
            ),
        )
        agent = _Agent(process, on_dialog=self._on_dialog)
        agent.start()
        try:
            state = await agent.call({"type": "get_state"}, timeout=60)
            reported = (state or {}).get("sessionId")
            if reported and reported != session_id:
                raise errors.WorkerError(
                    f"{KIND} opened session {reported}, not {session_id}"
                )
            # The approval gate must be there before any work is accepted:
            # a gate that failed to load is a gate that fails open.
            listed = await agent.call({"type": "get_commands"}, timeout=60)
            names = {
                c.get("name") for c in (listed or {}).get("commands") or []
            }
            if READY not in names:
                raise errors.WorkerError(
                    f"{KIND} did not load the SDK's extension ({EXTENSION}); "
                    "refusing to run without it"
                )
        except BaseException:
            await agent.close()
            with contextlib.suppress(Exception):
                await process.terminate()
            raise
        self._agent = agent

    async def _stage(self, history: list[messages_.Message]) -> str:
        """Write a session file with this past; start nothing on it."""
        session_id = str(uuid.uuid4())
        _handoff.warn_unsupported_files(history, KIND)
        dropped = _handoff.reasoning_count(history)
        if dropped:
            # pi drops thinking another model produced, and replaying it as
            # user text is refused by Anthropic's safeguards.
            warnings.warn(
                f"{dropped} reasoning part(s) in the history were not carried "
                "into pi: a model only reads its own reasoning. Text, tool "
                "calls and images were carried in full.",
                stacklevel=2,
            )
        directory = self._session_dir()
        now = datetime.datetime.now(datetime.UTC)
        stamp = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        lines: list[dict[str, Any]] = [
            {
                "type": "session",
                "version": 3,
                "id": session_id,
                "timestamp": stamp,
                "cwd": os.path.realpath(self._cwd),
            }
        ]
        parent: str | None = None
        # The model the conversation was on, as pi records a switch: pi
        # restores a session's model from its last reply's author. Measured:
        # without it, a seeded conversation ran on pi's default model.
        written: dict[str, Any] = next(
            (
                found
                for m in reversed(history)
                if m.role == "assistant"
                and isinstance(
                    found := (m.provider_metadata or {}).get(KIND), dict
                )
            ),
            {},
        )
        changes: list[dict[str, Any]] = []
        if written.get("provider") and written.get("model"):
            changes.append(
                {
                    "type": "model_change",
                    "provider": written["provider"],
                    "modelId": written["model"],
                }
            )
        if written.get("thinkingLevel"):
            changes.append(
                {
                    "type": "thinking_level_change",
                    "thinkingLevel": written["thinkingLevel"],
                }
            )
        entries = [
            *changes,
            *(
                {"type": "message", "message": message}
                for message in _handoff.to_pi_messages(
                    history, timestamp=int(now.timestamp() * 1000)
                )
            ),
        ]
        for entry in entries:
            entry_id = uuid.uuid4().hex[:8]
            lines.append(
                {
                    **entry,
                    "id": entry_id,
                    "parentId": parent,
                    "timestamp": stamp,
                }
            )
            parent = entry_id
        name = stamp.replace(":", "-").replace(".", "-")
        path = directory / f"{name}_{session_id}.jsonl"
        text = "".join(json.dumps(line) + "\n" for line in lines)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        await asyncio.to_thread(write)
        return session_id

    # -- approval -------------------------------------------------------------

    async def _on_dialog(self, agent: _Agent, request: dict[str, Any]) -> None:
        if request.get("method") not in DIALOGS:
            return  # a notice: nothing waits for an answer
        reply: dict[str, Any] = {
            "type": "extension_ui_response",
            "id": request.get("id"),
        }
        if (
            request.get("method") != "input"
            or request.get("title") != APPROVE
            or self._approve is None
            or self._interrupted
        ):
            # Another extension asking a person, who is not there; or the
            # turn is being stopped, and `abort` would wait on the question.
            agent.pending.pop(str(request.get("id")), None)
            await agent.send({**reply, "cancelled": True})
            return
        try:
            payload = json.loads(request.get("placeholder") or "{}")
        except json.JSONDecodeError:
            payload = {}
        granted, reason = await self._approve(
            str(payload.get("toolCallId") or ""),
            str(payload.get("toolName") or ""),
            dict(payload.get("input") or {}),
        )
        decision = (
            {"decision": "allow"}
            if granted
            else {"decision": "deny", "reason": reason}
        )
        # Answered from here on: `dismiss` must not withdraw it too.
        agent.pending.pop(str(request.get("id")), None)
        await agent.send({**reply, "value": json.dumps(decision)})

    # -- turns ----------------------------------------------------------------

    async def turn(self, prompt: base.Prompt) -> AsyncIterator[events_.Event]:
        base.check_files(prompt, KIND, ("image/",))
        agent = self._agent
        assert agent is not None
        if agent.dead is not None:
            raise errors.WorkerError(f"{KIND} crashed: {agent.dead}")
        # What the last turn left behind (a stopped turn's tail) is not this
        # turn's: `abort` answers only once pi is idle, so it has all arrived.
        while not agent.events.empty():
            agent.events.get_nowait()
        self._interrupted = False
        command = await _prompt_command(prompt)
        try:
            accepted = await agent.call(command)
        except _CommandError as exc:
            raise errors.turn_failed(KIND, str(exc), AUTH_HINT) from exc
        if (accepted or {}).get("disposition") == "handled":
            # An extension command consumed it. It may start a run of its own
            # (measured: then pi is already streaming, and that run is ours).
            state = await agent.call({"type": "get_state"})
            if not (state or {}).get("isStreaming"):
                yield events_.StreamStart()
                yield events_.StreamEnd(finish_reason="stop")
                return
        self._running = True
        usage = usage_.Usage()
        cost = 0.0
        started: set[str] = set()
        streamed: set[str] = set()
        open_blocks: dict[str, str] = {}
        message_no = 0
        last: dict[str, Any] = {}
        retry_error = ""
        settled: dict[str, Any] = {}
        try:
            yield events_.StreamStart()
            while True:
                event = await agent.events.get()
                kind = event.get("type")
                if kind == CRASHED:
                    raise errors.WorkerError(
                        f"{KIND} crashed: {event.get('detail')}"
                    )
                if kind == CLOSED:
                    raise errors.WorkerError(
                        f"{KIND}: the worker closed while its turn ran"
                    )
                message = event.get("message") or {}
                if kind == "message_start" and message.get("role") == (
                    "assistant"
                ):
                    message_no += 1
                elif kind == "message_update":
                    for out in _stream_events(
                        event.get("assistantMessageEvent") or {},
                        f"{message_no}",
                        open_blocks,
                        started,
                    ):
                        if isinstance(
                            out, events_.TextStart | events_.ReasoningStart
                        ):
                            streamed.add(out.block_id)
                        yield out
                elif kind == "message_end" and message.get("role") == (
                    "assistant"
                ):
                    last = message
                    raw = message.get("usage") or {}
                    usage = usage + _usage(raw)
                    cost += float((raw.get("cost") or {}).get("total") or 0)
                    for out in _close_blocks(open_blocks):
                        yield out
                    if message.get("stopReason") != "error":
                        # What the message holds that was never streamed (a
                        # run an extension command started, measured).
                        for out in _unstreamed(
                            message.get("content"),
                            f"{message_no}",
                            streamed,
                            started,
                        ):
                            yield out
                elif kind == "tool_execution_end" and not event.get(
                    "parentToolCallId"
                ):
                    # A nested call (a codemode script's) is not in the
                    # transcript: only the tool that made it reports it.
                    call_id = str(event.get("toolCallId") or "")
                    name = str(event.get("toolName") or "")
                    yield base.builtin_result(
                        call_id,
                        name,
                        _tool_result(event.get("result") or {}),
                        is_error=bool(event.get("isError")),
                    )
                elif kind == "auto_retry_end" and not event.get("success"):
                    retry_error = str(event.get("finalError") or "")
                elif kind == "agent_settled":
                    settled = event
                    break
        finally:
            self._running = False
        interrupted = self._interrupted or bool(settled.get("aborted"))
        self._interrupted = False
        stop = last.get("stopReason")
        if stop == "error" and not interrupted:
            raise errors.turn_failed(
                KIND,
                str(last.get("errorMessage") or retry_error),
                AUTH_HINT,
            )
        yield events_.StreamEnd(
            # Cost has no home in Usage; pi's own number rides in `raw`.
            usage=usage.model_copy(update={"raw": {"cost_usd": cost}}),
            finish_reason=(
                "cancelled"
                if interrupted or stop == "aborted"
                else "length"
                if stop == "length"
                else "stop"
            ),
            provider_metadata=_written_by(last),
        )

    async def steer(self, text: str) -> None:
        """Queue guidance for the running turn.

        pi delivers it once the current message's tool calls are done,
        before its next model call.
        """
        if not self._running or self._agent is None:
            raise errors.WorkerError(f"{KIND}: no turn is running to steer")
        try:
            await self._agent.call({"type": "steer", "message": text})
        except _CommandError as exc:
            raise errors.WorkerError(f"{KIND}: steer rejected: {exc}") from exc

    async def interrupt(self) -> None:
        self._interrupted = True
        agent = self._agent
        if agent is None or agent.dead is not None:
            return
        # A tool call waiting on approval keeps pi busy, and `abort` only
        # answers once pi is idle (measured: the turn ran on). Withdraw the
        # question first: the call is denied, and the abort lands at once.
        await agent.dismiss()
        with contextlib.suppress(Exception):
            await agent.call({"type": "abort"}, timeout=30)

    # -- the store ------------------------------------------------------------

    def _session_dir(self) -> pathlib.Path:
        """Where pi keeps this directory's sessions, decided as pi does.

        The environment, then project settings, then global ones; else one
        directory per working directory under the agent dir.
        """
        env = {**os.environ, **self._env}
        cwd = os.path.realpath(self._cwd)
        agent = pathlib.Path(
            os.path.expanduser(env.get("PI_CODING_AGENT_DIR") or "~/.pi/agent")
        )
        custom = (
            env.get("PI_CODING_AGENT_SESSION_DIR")
            or _setting(pathlib.Path(cwd, ".pi", "settings.json"))
            or _setting(agent / "settings.json")
        )
        if custom:
            return pathlib.Path(cwd, os.path.expanduser(custom))
        safe = cwd.lstrip("/\\")
        for c in "/\\:":
            safe = safe.replace(c, "-")
        return agent / "sessions" / f"--{safe}--"

    def _scan(self, *, titles: bool) -> list[dict[str, Any]]:
        """Describe the sessions of this directory, newest first."""
        directory = self._session_dir()
        cwd = os.path.realpath(self._cwd)
        out: list[dict[str, Any]] = []
        try:
            paths = sorted(directory.glob("*.jsonl"))
        except OSError:
            return []
        for path in paths:
            with contextlib.suppress(Exception):
                info = _describe(path, (cwd, self._cwd), titles=titles)
                if info is not None:
                    out.append(info)
        return sorted(out, key=lambda s: -s["updated"])

    async def _find(self, session_id: str) -> dict[str, Any]:
        if not SESSION_ID.match(session_id):
            raise errors.WorkerError(
                f"{KIND}: {session_id!r} is not a pi session id"
            )
        found = next(
            (
                s
                for s in await asyncio.to_thread(self._scan, titles=False)
                if s["id"] == session_id
            ),
            None,
        )
        if found is None:
            raise errors.WorkerError(
                f"{KIND}: no session {session_id} in {self._cwd}"
            )
        return found

    async def history(self, session_id: str) -> list[messages_.Message]:
        """Read the active branch of the session file.

        Every message on it, root to latest entry, including what a
        compaction summarized away, which pi keeps on disk.
        """
        found = await self._find(session_id)
        text = await asyncio.to_thread(pathlib.Path(found["path"]).read_text)
        stored: list[messages_.Message] = []
        for record in _active_branch(text):
            stored.extend(_stored_to_messages(record))
        return _handoff.model_view(stored)

    async def list_sessions(self) -> list[base.SessionInfo]:
        return [
            base.SessionInfo(
                session_id=s["id"],
                title=s.get("title"),
                cwd=s.get("cwd"),
                created_at=s.get("created"),
                updated_at=s.get("updated"),
            )
            for s in await asyncio.to_thread(self._scan, titles=True)
        ]

    def tui_command(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        running_as = session_id or str(uuid.uuid4())
        argv = [self._executable, "--session-id", running_as, *self._args()]
        servers = {
            k: v
            for k, v in self._mcp_servers.items()
            if k != _bridge.SERVER_NAME
        }
        # The person at the TUI is the gate: the extension asks nothing.
        return (
            argv,
            self._extension_env(approve=False, servers=servers),
            running_as,
        )


class _CommandError(Exception):
    """pi answered a command with `success: false`."""


class _Agent:
    """One `pi --mode rpc` process: its commands, its events, its dialogs.

    Responses resolve the command that asked (by id). Session events go on
    `events`, in order. A dialog is answered in a task of its own: an
    approval can take minutes, and the reader must keep reading meanwhile,
    including the reply to an `abort` sent from inside it.
    """

    def __init__(self, process: base.Process, *, on_dialog: Any) -> None:
        self.process = process
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.dead: str | None = None
        self._on_dialog = on_dialog
        self._waiters: dict[str, asyncio.Future[Any]] = {}
        self._pump: asyncio.Task[None] | None = None
        self._dialogs: set[asyncio.Task[None]] = set()
        # Dialogs pi still waits on, by request id: what `dismiss` withdraws.
        self.pending: dict[str, asyncio.Task[None]] = {}

    def start(self) -> None:
        self._pump = asyncio.create_task(self._read())

    async def call(
        self, command: dict[str, Any], *, timeout: float = 120
    ) -> Any:
        if self.dead is not None:
            raise errors.WorkerError(f"{KIND} crashed: {self.dead}")
        request_id = uuid.uuid4().hex
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = future
        try:
            await self.send({**command, "id": request_id})
            async with asyncio.timeout(timeout):
                return await future
        finally:
            self._waiters.pop(request_id, None)

    async def send(self, record: dict[str, Any]) -> None:
        try:
            await self.process.write(json.dumps(record) + "\n")
        except Exception as exc:
            raise errors.WorkerError(
                f"{KIND}: could not write to pi: {exc}"
            ) from exc

    def _track(self, request_id: str, task: asyncio.Task[None]) -> None:
        self.pending[request_id] = task

        def forget(_: asyncio.Task[None]) -> None:
            if self.pending.get(request_id) is task:
                del self.pending[request_id]

        task.add_done_callback(forget)

    async def dismiss(self) -> None:
        """Withdraw every question pi still waits on.

        The approval deciding it is cancelled, and pi hears `cancelled`: the
        extension then blocks the call.
        """
        for request_id, task in list(self.pending.items()):
            self.pending.pop(request_id, None)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            with contextlib.suppress(Exception):
                await self.send(
                    {
                        "type": "extension_ui_response",
                        "id": request_id,
                        "cancelled": True,
                    }
                )

    async def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
            self._pump = None
        for task in list(self._dialogs):
            task.cancel()
        for task in list(self._dialogs):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._dialogs.clear()
        self._fail("connection closed")
        # A turn reading events must hear it too: the producer is gone.
        self.events.put_nowait({"type": CLOSED})

    def _fail(self, detail: str) -> None:
        self.dead = self.dead or detail
        for future in self._waiters.values():
            if not future.done():
                future.set_exception(
                    errors.WorkerError(f"{KIND} crashed: {detail}")
                )
        self._waiters.clear()

    async def _read(self) -> None:
        try:
            while True:
                line = await self.process.readline()
                if not line:
                    detail = await self.process.death_note()
                    self._fail(detail)
                    self.events.put_nowait({"type": CRASHED, "detail": detail})
                    return
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # stdout is protocol only; tolerate noise
                if not isinstance(record, dict):
                    continue
                kind = record.get("type")
                if kind == "response":
                    self._resolve(record)
                elif kind == "extension_ui_request":
                    task = asyncio.create_task(self._on_dialog(self, record))
                    self._dialogs.add(task)
                    task.add_done_callback(self._dialogs.discard)
                    if record.get("method") in DIALOGS:
                        self._track(str(record.get("id")), task)
                else:
                    self.events.put_nowait(record)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(str(exc))
            self.events.put_nowait({"type": CRASHED, "detail": str(exc)})

    def _resolve(self, record: dict[str, Any]) -> None:
        future = self._waiters.get(str(record.get("id")))
        if future is None or future.done():
            return
        if record.get("success") is False:
            future.set_exception(_CommandError(str(record.get("error") or "")))
        else:
            future.set_result(record.get("data"))


async def _prompt_command(prompt: base.Prompt) -> dict[str, Any]:
    """Return a prompt as pi's `prompt` command: text, and images as base64."""
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for part in prompt:
        if isinstance(part, messages_.TextPart):
            texts.append(part.text)
        elif part.media_type.startswith("text/"):
            texts.append(await base.as_text(part))
        else:
            images.append(
                {
                    "type": "image",
                    "data": base64.b64encode(await base.data(part)).decode(
                        "ascii"
                    ),
                    "mimeType": part.media_type,
                }
            )
    command: dict[str, Any] = {"type": "prompt", "message": "\n\n".join(texts)}
    if images:
        command["images"] = images
    return command


def _stream_events(
    update: dict[str, Any],
    message: str,
    open_blocks: dict[str, str],
    started: set[str],
) -> list[events_.Event]:
    """One `assistantMessageEvent` -> events.

    Block ids are `<message>.<contentIndex>`: pi numbers content blocks per
    assistant message, and a turn holds several. Tool-call arguments travel
    as ONE delta, the final arguments, at `toolcall_end`: streamed fragments
    are not guaranteed to add up to what pi settles on.
    """
    kind = update.get("type") or ""
    block = f"{message}.{update.get('contentIndex', 0)}"
    out: list[events_.Event] = []
    if kind == "text_start":
        open_blocks[block] = "text"
        out.append(events_.TextStart(block_id=block))
    elif kind == "text_delta" and update.get("delta"):
        out.append(events_.TextDelta(chunk=update["delta"], block_id=block))
    elif kind == "thinking_start":
        open_blocks[block] = "thinking"
        out.append(events_.ReasoningStart(block_id=block))
    elif kind == "thinking_delta" and update.get("delta"):
        out.append(
            events_.ReasoningDelta(chunk=update["delta"], block_id=block)
        )
    elif kind in ("text_end", "thinking_end"):
        out += _close_blocks(open_blocks, only=block)
    elif kind == "toolcall_start" and update.get("id"):
        started.add(update["id"])
        out.append(
            events_.BuiltinToolStart(
                tool_call_id=update["id"],
                tool_name=update.get("toolName") or "",
            )
        )
    elif kind == "toolcall_end":
        call = _tool_call(update.get("toolCall") or {})
        out += base.builtin_call(
            call.tool_call_id, call.tool_name, call.tool_args, started=started
        )
    return out


def _unstreamed(
    content: Any, message: str, streamed: set[str], started: set[str]
) -> list[events_.Event]:
    """Return the events of a finished message's unstreamed blocks."""
    out: list[events_.Event] = []
    blocks = content if isinstance(content, list) else []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        block_id = f"{message}.{index}"
        kind = block.get("type")
        if kind == "text" and block.get("text") and block_id not in streamed:
            out += [
                events_.TextStart(block_id=block_id),
                events_.TextDelta(chunk=block["text"], block_id=block_id),
                events_.TextEnd(block_id=block_id),
            ]
        elif (
            kind == "thinking"
            and block.get("thinking")
            and block_id not in streamed
        ):
            out += [
                events_.ReasoningStart(block_id=block_id),
                events_.ReasoningDelta(
                    chunk=block["thinking"], block_id=block_id
                ),
                events_.ReasoningEnd(block_id=block_id),
            ]
        elif kind == "toolCall" and str(block.get("id")) not in started:
            call = _tool_call(block)
            out += base.builtin_call(
                call.tool_call_id,
                call.tool_name,
                call.tool_args,
                started=started,
            )
    return out


def _close_blocks(
    open_blocks: dict[str, str], only: str | None = None
) -> list[events_.Event]:
    """End the text and reasoning blocks still open (an aborted message's)."""
    out: list[events_.Event] = []
    for block in [only] if only is not None else list(open_blocks):
        opened = open_blocks.pop(block, None)
        if opened == "text":
            out.append(events_.TextEnd(block_id=block))
        elif opened == "thinking":
            out.append(events_.ReasoningEnd(block_id=block))
    return out


def _tool_call(block: dict[str, Any]) -> messages_.ToolCallPart:
    return messages_.ToolCallPart(
        tool_call_id=str(block.get("id") or ""),
        tool_name=str(block.get("name") or ""),
        tool_args=json.dumps(block.get("arguments") or {}),
    )


def _assistant_parts(content: Any) -> list[messages_.Part]:
    parts: list[messages_.Part] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            parts.append(messages_.TextPart(text=block["text"]))
        elif kind == "thinking" and block.get("thinking"):
            parts.append(messages_.ReasoningPart(text=block["thinking"]))
        elif kind == "toolCall":
            parts.append(_tool_call(block))
    return parts


def _content_parts(content: Any) -> list[messages_.Part]:
    """Return a user message's parts: from a string, or text and images."""
    if isinstance(content, str):
        return [messages_.TextPart(text=content)] if content else []
    parts: list[messages_.Part] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            parts.append(messages_.TextPart(text=block["text"]))
        elif block.get("type") == "image" and block.get("data"):
            parts.append(
                messages_.FilePart(
                    data=str(block["data"]),
                    media_type=str(block.get("mimeType") or "image/png"),
                )
            )
    return parts


def _tool_result(result: Any) -> Any:
    """Return what a tool returned: its text, when that is all it is.

    pi tools return content blocks; text-only content (nearly every call)
    reads as the string it is, anything else stays the blocks themselves.
    """
    content = result.get("content") if isinstance(result, dict) else result
    if isinstance(content, list) and all(
        isinstance(b, dict) and b.get("type") == "text" for b in content
    ):
        content = "".join(b.get("text") or "" for b in content)
    return content


def _active_branch(text: str) -> list[dict[str, Any]]:
    """Return the session file's active branch: root to latest entry.

    A session is a tree (`id`/`parentId`); the leaf is the entry appended
    last, and abandoned branches are not part of the conversation.
    """
    entries: dict[str, dict[str, Any]] = {}
    leaf: str | None = None
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(entry, dict)
            and entry.get("id")
            and entry.get("type") != "session"
        ):
            entries[str(entry["id"])] = entry
            leaf = str(entry["id"])
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    while leaf is not None and leaf in entries and leaf not in seen:
        seen.add(leaf)
        path.append(entries[leaf])
        parent = entries[leaf].get("parentId")
        leaf = str(parent) if parent is not None else None
    return list(reversed(path))


def _stored_to_messages(entry: dict[str, Any]) -> list[messages_.Message]:
    """One session entry -> messages, CLI tools as tool calls.

    Everything pi sends the model is kept: what an extension injected
    (`custom_message`) and a `!` command's output (`bashExecution`) reach it
    as user messages, and so they are user messages here; a branch summary
    is the context pi carried over from an abandoned branch.
    """
    kind = entry.get("type")
    if kind == "custom_message":
        parts = _content_parts(entry.get("content"))
        return [messages_.Message(role="user", parts=parts)] if parts else []
    if kind == "branch_summary" and entry.get("summary"):
        text = str(entry["summary"])
        return [
            messages_.Message(
                role="user", parts=[messages_.TextPart(text=text)]
            )
        ]
    if kind != "message":
        return []
    message = entry.get("message") or {}
    role = message.get("role")
    if role in ("custom", "user"):
        parts = _content_parts(message.get("content"))
        return [messages_.Message(role="user", parts=parts)] if parts else []
    if role == "bashExecution" and not message.get("excludeFromContext"):
        # pi's own rendering of it for the model.
        text = (
            f"Ran `{message.get('command', '')}`\n```\n"
            f"{message.get('output', '')}\n```"
        )
        return [
            messages_.Message(
                role="user", parts=[messages_.TextPart(text=text)]
            )
        ]
    if role == "assistant":
        parts = _assistant_parts(message.get("content"))
        if not parts:
            return []
        return [
            messages_.Message(
                role="assistant",
                parts=parts,
                provider_metadata=_written_by(message),
            )
        ]
    if role == "toolResult":
        result = messages_.ToolResultPart(
            tool_call_id=str(message.get("toolCallId") or ""),
            tool_name=str(message.get("toolName") or ""),
            result=_tool_result(message),
            result_kind="error" if message.get("isError") else "json",
        )
        return [messages_.Message(role="tool", parts=[result])]
    return []


def _written_by(message: dict[str, Any]) -> dict[str, Any] | None:
    """Which provider and model wrote an assistant message, as pi says.

    Kept on the message (`provider_metadata["pi"]`) so a conversation
    seeded into another pi continues on the model it was on.
    """
    written = {
        k: message[k]
        for k in ("api", "provider", "model", "thinkingLevel")
        if isinstance(message.get(k), str)
    }
    if not written or written.get("provider") == "ai-python":
        return None
    return {KIND: written}


def _usage(raw: dict[str, Any]) -> usage_.Usage:
    """Return pi's per-message usage. pi's `input` excludes cached input."""

    def count(name: str) -> int:
        value = raw.get(name)
        return value if isinstance(value, int) else 0

    reasoning = raw.get("reasoning")
    return usage_.Usage(
        input_tokens=count("input") + count("cacheRead") + count("cacheWrite"),
        output_tokens=count("output"),
        cache_read_tokens=count("cacheRead"),
        cache_write_tokens=count("cacheWrite"),
        reasoning_tokens=reasoning if isinstance(reasoning, int) else None,
    )


def mcp_config(servers: dict[str, Any]) -> dict[str, Any]:
    """MCP servers, in the shape claude takes, as pi registers them.

    pi validates the config, so only the keys it knows are passed.
    """
    keys = ("command", "args", "env", "cwd", "url", "headers")
    return {
        name: {k: server[k] for k in keys if server.get(k) is not None}
        for name, server in servers.items()
    }


def _setting(path: pathlib.Path) -> str | None:
    """Return the `sessionDir` a pi settings file names, if any."""
    try:
        value = json.loads(path.read_text()).get("sessionDir")
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _describe(
    path: pathlib.Path, cwds: tuple[str, ...], *, titles: bool
) -> dict[str, Any] | None:
    """Describe one session file of this directory, or None.

    A session file can come from anyone (a project's settings choose the
    directory), and its id becomes a CLI argument: one that breaks pi's
    rule is not a session.
    """
    with path.open() as f:
        header = json.loads(f.readline() or "{}")
        if header.get("type") != "session" or not SESSION_ID.match(
            str(header.get("id") or "")
        ):
            return None
        if header.get("cwd") not in cwds:
            return None
        created: int | None = None
        with contextlib.suppress(ValueError):
            when = datetime.datetime.strptime(
                str(header.get("timestamp") or "")[:19], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=datetime.UTC)
            created = int(when.timestamp() * 1000)
        info: dict[str, Any] = {
            "id": header["id"],
            "path": str(path),
            "cwd": header.get("cwd"),
            "created": created,
            "updated": int(path.stat().st_mtime * 1000),
        }
        if not titles:
            return info
        name: str | None = None
        first: str | None = None
        for line in f:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("type") == "session_info":
                name = entry.get("name") or None
            elif first is None and entry.get("type") == "message":
                message = entry.get("message") or {}
                if message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    first = (content or "").strip()[:200] or None
        info["title"] = name or first
        return info
