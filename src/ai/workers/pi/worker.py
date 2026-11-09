"""Pi, driven through `pi --mode rpc`.

One pi process per session: an RPC process holds one session. The id we
hand pi (`--session-id`) IS its session id, so `pi --session-id <id>` opens
the same conversation in pi's own TUI.

Every pi a worker starts loads `extension.js`: pi has no flag for an MCP
server, so the extension registers the worker's MCP bridge, and a marker
command it registers last proves that it loaded.

The protocol shapes here were measured against a real pi (see
VERIFIED_VERSION), not taken from the docs alone.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import datetime
import json
import os
import pathlib
import re
import uuid
import warnings
from collections.abc import AsyncGenerator, Iterator
from typing import Any, Literal

import pydantic

from ...types import events as events_
from ...types import messages as messages_
from ...types import usage as usage_
from .. import _bridge, _convert, _handoff, _process, _wire, _worker
from .. import _errors as errors
from . import handoff

KIND = "pi"
INSTALL = "npm install -g @earendil-works/pi-coding-agent"
# The CLI version the live tests last passed with. Older ones get a warning.
VERIFIED_VERSION = "1.5.0"
EXTENSION = str(pathlib.Path(__file__).with_name("extension.js"))
# The command the extension registers last: proof that it loaded.
READY = "ai-python-ready"
# The environment variable that configures the extension.
CONFIG_VAR = "AI_PYTHON_PI"
# pi names an MCP tool `mcp__<server>__<tool>` (with a hash suffix when the
# name is longer than 64 characters; the prefix stays).
BRIDGE_PREFIX = f"mcp__{_bridge.SERVER_NAME}__"
# pi's rule for a session id: letters, digits, `.`, `_` and `-`, starting and
# ending with a letter or digit. Never a path.
SESSION_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,198}[A-Za-z0-9])?\Z")
# Extension UI methods that wait for an answer; the rest are notices.
DIALOGS = ("select", "confirm", "input", "editor")

AUTH_HINT = (
    "pi uses its own login (`/login`, auth.json) or a provider key in the "
    "environment: pass env={'ANTHROPIC_API_KEY': ...}."
)


class Pi(_worker.Worker):
    """Pi, over `pi --mode rpc`.

    `provider` picks the provider of `model_name` (`--provider`), and
    `thinking` the thinking level. `allowed_tools` limits pi's own tools
    (`["read", "grep"]`, or `[]` for none; `ai` tools stay), and
    `disallowed_tools` takes some away. `args` go to pi as they are. pi never
    asks for approval.
    """

    worker_class_id: Literal["pi"] = "pi"
    provider: str | None = None
    thinking: (
        Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
        | None
    ) = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    args: list[str] = pydantic.Field(default_factory=list)

    def _make_wire(self) -> _PiWire:
        return _PiWire(self)


class _CommandError(Exception):
    """pi answered a command with `success: false`."""


class _PiWire:
    def __init__(self, worker: Pi) -> None:
        self.worker = worker
        # pi records a session's directory as its real path (on macOS a temp
        # dir /tmp/... is recorded as /private/tmp/..., measured).
        self.cwd = os.path.realpath(worker.cwd)
        self.process: _process.Process | None = None
        self.reader: asyncio.Task[None] | None = None
        self.dead: str | None = None
        self.waiters: dict[str, asyncio.Future[Any]] = {}
        # Unbounded: the reader fills it inline and must not wait.
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        # User messages that pi took but did not put in the conversation yet,
        # in the order sent. See `send`.
        self.pending: collections.deque[messages_.Message] = collections.deque()
        # Between `agent_start` and `agent_settled`, as the reader saw it.
        self.streaming = False
        # Assistant messages seen in the session: block ids are unique.
        self.count = 0

    # -- life -----------------------------------------------------------------

    async def start(
        self,
        *,
        bridge_url: str,
        instructions: str | None,
        seed: list[messages_.Message],
    ) -> str:
        worker = self.worker
        executable = worker.executable or "pi"
        await _wire.probe(
            KIND,
            executable,
            install=INSTALL,
            verified=VERIFIED_VERSION,
            cwd=self.cwd,
            env=worker.env,
        )
        session_id = worker.session_id
        if session_id is not None:
            # Checked first: `--session-id` CREATES a session it cannot
            # find, so a typo would silently start an empty conversation.
            await asyncio.to_thread(self._find, session_id)
        else:
            session_id = str(uuid.uuid4())
            if seed:
                await asyncio.to_thread(self._stage, session_id, seed)
        argv = [
            executable,
            "--mode",
            "rpc",
            "--session-id",
            session_id,
            "-e",
            EXTENSION,
        ]
        if worker.model_name:
            argv += ["--model", worker.model_name]
        if worker.provider:
            argv += ["--provider", worker.provider]
        if worker.thinking:
            argv += ["--thinking", worker.thinking]
        if instructions:
            argv += ["--append-system-prompt", instructions]
        if worker.allowed_tools is not None:
            # Not `--no-tools`: it disables extension tools too, and the
            # MCP bridge's tools are extension tools.
            argv += (
                ["--tools", ",".join(worker.allowed_tools)]
                if worker.allowed_tools
                else ["--no-builtin-tools"]
            )
        if worker.disallowed_tools:
            argv += ["--exclude-tools", ",".join(worker.disallowed_tools)]
        argv += worker.args
        bridge = {
            "url": bridge_url,
            # pi's default exposure is "codemode": the model reaches MCP
            # tools through a script, as nested calls. "direct" declares
            # each tool to the model, and the first prompt waits until the
            # bridge is connected.
            "exposure": "direct",
            # An `ai` tool can take longer than pi's default (60 seconds).
            "timeout": 24 * 3600,
        }
        config = {"mcp": {_bridge.SERVER_NAME: bridge}}
        self.process = await _process.spawn(
            argv,
            cwd=self.cwd,
            env={**worker.env, CONFIG_VAR: json.dumps(config)},
        )
        self.reader = asyncio.create_task(self._read(self.process))
        try:
            # A pi that cannot start (an unknown model, a bad flag) exits at
            # once, and these calls report its last words.
            state = await self._call({"type": "get_state"}, timeout=60)
            reported = (state or {}).get("sessionId")
            if reported and reported != session_id:
                raise errors.WorkerError(
                    f"{KIND} opened session {reported}, not {session_id}"
                )
            listed = await self._call({"type": "get_commands"}, timeout=60)
            names = {
                c.get("name") for c in (listed or {}).get("commands") or []
            }
            if READY not in names:
                raise errors.WorkerError(
                    f"{KIND} did not load the SDK's extension ({EXTENSION}); "
                    "without it pi has no `ai` tools"
                )
        except BaseException:
            # A failed start must not leave pi running.
            await self.close()
            raise
        return session_id

    async def close(self) -> None:
        reader, self.reader = self.reader, None
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
        self._fail("closed")
        process, self.process = self.process, None
        if process is not None:
            with contextlib.suppress(Exception):
                await process.terminate()

    # -- input ----------------------------------------------------------------

    async def send(self, message: messages_.Message) -> None:
        _convert.check_parts(message, KIND, ("image/",))
        command = await _prompt_command(message)
        # `Accepted` comes from pi's own record of the message: the
        # `message_start` of a user message, which pi emits when the message
        # enters the conversation and before anything the message causes
        # (measured, for a prompt and for a steer). The command response
        # does not order anything: for a new run, it can come after the
        # run's records; for a steer, it comes long before the message is
        # delivered. So the message waits in `pending`, registered BEFORE
        # the write, and the reader matches user messages to it in order.
        self.pending.append(message)
        try:
            # Always `prompt` with `steer`: when pi is idle it starts a run,
            # when a run streams it queues the message for the next model
            # call. pi chooses atomically, so a send at the very end of a run
            # is never stranded (`steer` alone queues even when idle, and
            # nothing then delivers it). Measured: "started" when idle,
            # "queued" while streaming. A queued steer is delivered before
            # the run settles, also after `abort`.
            accepted = await self._call(
                {**command, "streamingBehavior": "steer"}
            )
        except _CommandError as exc:
            # Rejected before acceptance: pi will never report it.
            # Measured: pi refuses a prompt itself when it has no key.
            self._forget(message)
            raise errors.turn_failed(KIND, str(exc), AUTH_HINT) from exc
        disposition = (accepted or {}).get("disposition")
        if disposition == "handled":
            # An extension command or input handler consumed it: no user
            # message comes. A command may start a run of its own (measured:
            # then pi already streams, and that run follows); if nothing
            # runs, the turn is over.
            self._forget(message)
            state = await self._call({"type": "get_state"})
            self.queue.put_nowait(("accepted", message))
            if not (state or {}).get("isStreaming") and not self.streaming:
                self.queue.put_nowait(("handled", None))
        elif disposition not in ("started", "queued"):
            # An ambiguous acceptance must never be retried as a new turn.
            raise errors.WorkerError(
                f"{KIND}: unknown prompt disposition: {accepted}"
            )

    def _forget(self, message: messages_.Message) -> None:
        with contextlib.suppress(ValueError):
            self.pending.remove(message)

    async def interrupt(self) -> None:
        if self.process is None or self.dead is not None:
            return
        # `abort` answers only after the run settled (measured: the aborted
        # message, `agent_end` and `agent_settled {aborted: true}` come
        # first). So it is sent, not awaited: the end comes as a TurnEnd.
        with contextlib.suppress(Exception):
            await self._write({"type": "abort", "id": uuid.uuid4().hex})

    # -- from pi --------------------------------------------------------------

    async def _call(
        self, command: dict[str, Any], *, timeout: float = 120
    ) -> Any:
        if self.dead is not None:
            raise errors.WorkerError(f"{KIND} crashed: {self.dead}")
        request_id = uuid.uuid4().hex
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.waiters[request_id] = future
        try:
            await self._write({**command, "id": request_id})
            async with asyncio.timeout(timeout):
                return await future
        finally:
            self.waiters.pop(request_id, None)

    async def _write(self, record: dict[str, Any]) -> None:
        if self.process is None:
            raise errors.WorkerError(f"{KIND}: pi is not running")
        try:
            await self.process.write(json.dumps(record) + "\n")
        except Exception as exc:
            raise errors.WorkerError(
                f"{KIND}: could not write to pi: {exc}"
            ) from exc

    def _fail(self, detail: str) -> None:
        self.dead = self.dead or detail
        for future in self.waiters.values():
            if not future.done():
                future.set_exception(
                    errors.WorkerError(f"{KIND} crashed: {detail}")
                )
        self.waiters.clear()

    async def _read(self, process: _process.Process) -> None:
        try:
            while True:
                line = await process.readline()
                if not line:
                    detail = await process.death_note()
                    self._fail(detail)
                    self.queue.put_nowait(("crashed", detail))
                    return
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # stdout is protocol only; tolerate noise
                if not isinstance(record, dict):
                    continue
                kind = record.get("type")
                if kind == "response":
                    future = self.waiters.get(str(record.get("id")))
                    if future is None or future.done():
                        continue
                    if record.get("success") is False:
                        future.set_exception(
                            _CommandError(str(record.get("error") or ""))
                        )
                    else:
                        future.set_result(record.get("data"))
                elif kind == "extension_ui_request":
                    if record.get("method") in DIALOGS:
                        # Another extension asks a person, who is not there
                        # (DESIGN.md D1). Unanswered, the agent would wait.
                        await self._write(
                            {
                                "type": "extension_ui_response",
                                "id": record.get("id"),
                                "cancelled": True,
                            }
                        )
                else:
                    if kind == "agent_start":
                        self.streaming = True
                    elif kind == "agent_settled":
                        self.streaming = False
                    self.queue.put_nowait(("record", record))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(str(exc))
            self.queue.put_nowait(("crashed", str(exc)))

    async def events(self) -> AsyncGenerator[_wire.Item]:
        run = _Run()
        while True:
            kind, value = await self.queue.get()
            if kind == "crashed":
                raise errors.WorkerError(f"{KIND} crashed: {value}")
            if kind == "accepted":
                yield _wire.Accepted(value)
            elif kind == "handled":
                yield _wire.TurnEnd()
            else:
                if value.get("type") == "agent_start":
                    run = _Run()
                for item in self._translate(value, run):
                    yield item

    def _translate(
        self, record: dict[str, Any], run: _Run
    ) -> Iterator[_wire.Item]:
        kind = record.get("type")
        message = record.get("message") or {}
        role = message.get("role")
        if kind == "message_start" and role == "user":
            # pi's own system message (`role: "system"`, pi 1.5) and other
            # roles are not ours. A user message nobody sent here (an
            # extension's) has nothing pending.
            if self.pending:
                yield _wire.Accepted(self.pending.popleft())
        elif kind == "message_start" and role == "assistant":
            self.count += 1
        elif kind == "message_update":
            yield from _stream_events(
                record.get("assistantMessageEvent") or {}, str(self.count), run
            )
        elif kind == "message_end" and role == "assistant":
            run.last = message
            raw = message.get("usage") or {}
            run.usage = run.usage + _usage(raw)
            run.cost += float((raw.get("cost") or {}).get("total") or 0)
            yield from _close_blocks(run)
            if message.get("stopReason") != "error":
                # What the message holds that was never streamed (a run an
                # extension command started, measured).
                yield from _unstreamed(
                    message.get("content"), str(self.count), run
                )
        elif kind == "tool_execution_end":
            call_id = str(record.get("toolCallId") or "")
            name = str(record.get("toolName") or "")
            # A nested call (a codemode script's) is not in the transcript:
            # only the tool that made it reports it. A bridge call is the
            # worker's to report.
            if (
                not record.get("parentToolCallId")
                and call_id not in run.hidden
                and not name.startswith(BRIDGE_PREFIX)
            ):
                yield _convert.builtin_result(
                    call_id,
                    name,
                    _tool_result(record.get("result") or {}),
                    is_error=bool(record.get("isError")),
                )
        elif kind == "auto_retry_end" and not record.get("success"):
            run.retry_error = str(record.get("finalError") or "")
        elif kind == "agent_settled":
            # `agent_end` closes one low-level run, but retries, compaction
            # and queued steers can follow it. `agent_settled` means pi will
            # not continue on its own: the turn is over.
            yield from _close_blocks(run)
            stop = run.last.get("stopReason")
            usage = run.usage.model_copy(update={"raw": {"cost_usd": run.cost}})
            written = _written_by(run.last)
            if record.get("aborted") or stop == "aborted":
                yield _wire.TurnEnd(
                    reason="interrupted", usage=usage, metadata=written
                )
            elif stop == "error":
                error = errors.turn_failed(
                    KIND,
                    str(run.last.get("errorMessage") or run.retry_error),
                    AUTH_HINT,
                )
                yield _wire.TurnEnd(reason="error", usage=usage, error=error)
            else:
                reason: Literal["stop", "length"] = (
                    "length" if stop == "length" else "stop"
                )
                yield _wire.TurnEnd(
                    reason=reason, usage=usage, metadata=written
                )

    # -- the store ------------------------------------------------------------

    def _session_dir(self) -> pathlib.Path:
        """Where pi keeps this directory's sessions, decided as pi does.

        The environment, then project settings, then global ones; else one
        directory per working directory under the agent dir.
        """
        env = {**os.environ, **self.worker.env}
        agent = pathlib.Path(
            os.path.expanduser(env.get("PI_CODING_AGENT_DIR") or "~/.pi/agent")
        )
        custom = (
            env.get("PI_CODING_AGENT_SESSION_DIR")
            or _setting(pathlib.Path(self.cwd, ".pi", "settings.json"))
            or _setting(agent / "settings.json")
        )
        if custom:
            return pathlib.Path(self.cwd, os.path.expanduser(custom))
        safe = self.cwd.lstrip("/\\")
        for c in "/\\:":
            safe = safe.replace(c, "-")
        return agent / "sessions" / f"--{safe}--"

    def _find(self, session_id: str) -> None:
        """Refuse a session that is not this directory's.

        A session file can come from anyone (a project's settings choose the
        directory), and its id becomes a CLI argument: one that breaks pi's
        rule is not a session.
        """
        if not SESSION_ID.match(session_id):
            raise errors.WorkerError(
                f"{KIND}: {session_id!r} is not a pi session id"
            )
        try:
            paths = sorted(self._session_dir().glob("*.jsonl"))
        except OSError:
            paths = []
        for path in paths:
            try:
                with path.open() as f:
                    header = json.loads(f.readline() or "{}")
            except Exception:
                continue
            if (
                not isinstance(header, dict)
                or header.get("type") != "session"
                or header.get("id") != session_id
            ):
                continue
            cwd = header.get("cwd")
            if cwd not in (self.cwd, self.worker.cwd):
                raise errors.WorkerError(
                    f"{KIND}: session {session_id} belongs to {cwd}, "
                    f"not {self.cwd}"
                )
            return
        raise errors.WorkerError(
            f"{KIND}: no session {session_id} in {self.cwd}"
        )

    def _stage(self, session_id: str, seed: list[messages_.Message]) -> None:
        """Write a session file with this past; start nothing on it."""
        _handoff.warn_unsupported_files(seed, KIND)
        dropped = _handoff.reasoning_count(seed)
        if dropped:
            # pi drops thinking another model produced, and replaying it as
            # user text is refused by Anthropic's safeguards.
            warnings.warn(
                f"{dropped} reasoning part(s) in the history were not carried "
                "into pi: a model only reads its own reasoning. Text, tool "
                "calls and images were carried in full.",
                stacklevel=2,
            )
        now = datetime.datetime.now(datetime.UTC)
        stamp = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        lines: list[dict[str, Any]] = [
            {
                "type": "session",
                "version": 3,
                "id": session_id,
                "timestamp": stamp,
                "cwd": self.cwd,
            }
        ]
        # The model the conversation was on, as pi records a switch: pi
        # restores a session's model from its last reply's author. Measured:
        # without it, a seeded conversation ran on pi's default model. The
        # wire records the author of each turn's last reply in
        # `provider_metadata["pi"]` (`_written_by`).
        written: dict[str, Any] = next(
            (
                found
                for m in reversed(seed)
                if m.role == "assistant"
                and isinstance(
                    found := (m.provider_metadata or {}).get(KIND), dict
                )
            ),
            {},
        )
        entries: list[dict[str, Any]] = []
        if written.get("provider") and written.get("model"):
            entries.append(
                {
                    "type": "model_change",
                    "provider": written["provider"],
                    "modelId": written["model"],
                }
            )
        if written.get("thinkingLevel"):
            entries.append(
                {
                    "type": "thinking_level_change",
                    "thinkingLevel": written["thinkingLevel"],
                }
            )
        entries += [
            {"type": "message", "message": message}
            for message in handoff.to_pi_messages(
                seed, timestamp=int(now.timestamp() * 1000)
            )
        ]
        parent: str | None = None
        for entry in entries:
            # A chain, root first: pi reads it as one branch.
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
        path = self._session_dir() / f"{name}_{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))


class _Run:
    """What one pi run's translation remembers."""

    def __init__(self) -> None:
        # Open text and thinking blocks, by block id: "text" or "thinking".
        self.open: dict[str, str] = {}
        self.streamed: set[str] = set()
        self.started: set[str] = set()
        # Bridge calls: the worker reports them, not the wire.
        self.hidden: set[str] = set()
        self.last: dict[str, Any] = {}
        self.usage = usage_.Usage()
        self.cost = 0.0
        # The last error pi reported while it retried: if the run fails
        # without a message of its own, the only explanation.
        self.retry_error = ""


async def _prompt_command(message: messages_.Message) -> dict[str, Any]:
    """Return a user message as pi's `prompt`: text, and images as base64."""
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, messages_.TextPart):
            texts.append(part.text)
        elif isinstance(part, messages_.FilePart):
            if part.media_type.startswith("text/"):
                texts.append(await _convert.as_text(part))
            else:
                encoded = base64.b64encode(await _convert.data(part))
                images.append(
                    {
                        "type": "image",
                        "data": encoded.decode("ascii"),
                        "mimeType": part.media_type,
                    }
                )
    command: dict[str, Any] = {"type": "prompt", "message": "\n\n".join(texts)}
    if images:
        command["images"] = images
    return command


def _stream_events(
    update: dict[str, Any], message: str, run: _Run
) -> list[_wire.Content]:
    """One `assistantMessageEvent` -> events.

    Block ids are `<message>.<contentIndex>`: pi numbers content blocks per
    assistant message, and a turn holds several. Tool-call arguments travel
    as ONE delta, the final arguments, at `toolcall_end`: streamed fragments
    are not guaranteed to add up to what pi settles on.
    """
    kind = update.get("type") or ""
    block = f"{message}.{update.get('contentIndex', 0)}"
    out: list[_wire.Content] = []
    if kind == "text_start":
        run.open[block] = "text"
        run.streamed.add(block)
        out.append(events_.TextStart(block_id=block))
    elif kind == "text_delta" and update.get("delta"):
        out.append(events_.TextDelta(chunk=update["delta"], block_id=block))
    elif kind == "thinking_start":
        run.open[block] = "thinking"
        run.streamed.add(block)
        out.append(events_.ReasoningStart(block_id=block))
    elif kind == "thinking_delta" and update.get("delta"):
        out.append(
            events_.ReasoningDelta(chunk=update["delta"], block_id=block)
        )
    elif kind in ("text_end", "thinking_end"):
        out += _close_blocks(run, only=block)
    elif kind == "toolcall_start" and update.get("id"):
        call_id = str(update["id"])
        name = str(update.get("toolName") or "")
        if name.startswith(BRIDGE_PREFIX):
            run.hidden.add(call_id)
        else:
            run.started.add(call_id)
            out.append(
                events_.BuiltinToolStart(tool_call_id=call_id, tool_name=name)
            )
    elif kind == "toolcall_end":
        out += _builtin_call(update.get("toolCall") or {}, run)
    return out


def _unstreamed(
    content: Any, message: str, run: _Run
) -> Iterator[_wire.Content]:
    """Yield the events of a finished message's unstreamed blocks."""
    for index, block in enumerate(content if isinstance(content, list) else []):
        if not isinstance(block, dict):
            continue
        block_id = f"{message}.{index}"
        kind = block.get("type")
        if (
            kind == "text"
            and block.get("text")
            and block_id not in run.streamed
        ):
            yield events_.TextStart(block_id=block_id)
            yield events_.TextDelta(chunk=block["text"], block_id=block_id)
            yield events_.TextEnd(block_id=block_id)
        elif (
            kind == "thinking"
            and block.get("thinking")
            and block_id not in run.streamed
        ):
            yield events_.ReasoningStart(block_id=block_id)
            yield events_.ReasoningDelta(
                chunk=block["thinking"], block_id=block_id
            )
            yield events_.ReasoningEnd(block_id=block_id)
        elif kind == "toolCall" and str(block.get("id")) not in run.started:
            yield from _builtin_call(block, run)


def _builtin_call(block: dict[str, Any], run: _Run) -> list[_wire.Content]:
    """Return the events of a finished pi tool call; none for a bridge's."""
    call_id = str(block.get("id") or "")
    name = str(block.get("name") or "")
    if call_id in run.hidden or name.startswith(BRIDGE_PREFIX):
        run.hidden.add(call_id)
        return []
    args = json.dumps(block.get("arguments") or {})
    return list(_convert.builtin_call(call_id, name, args, started=run.started))


def _close_blocks(run: _Run, only: str | None = None) -> list[_wire.Content]:
    """End the text and reasoning blocks still open (an aborted message's)."""
    out: list[_wire.Content] = []
    for block in [only] if only is not None else list(run.open):
        opened = run.open.pop(block, None)
        if opened == "text":
            out.append(events_.TextEnd(block_id=block))
        elif opened == "thinking":
            out.append(events_.ReasoningEnd(block_id=block))
    return out


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


def _setting(path: pathlib.Path) -> str | None:
    """Return the `sessionDir` a pi settings file names, if any."""
    try:
        value = json.loads(path.read_text()).get("sessionDir")
    except Exception:
        return None
    return value if isinstance(value, str) and value else None


def _written_by(message: dict[str, Any]) -> dict[str, Any] | None:
    """Say which provider and model wrote an assistant message, as pi does.

    Kept on the message (`provider_metadata["pi"]`) so a conversation
    seeded into another pi continues on the model it was on.
    """
    written = {
        k: message[k]
        for k in ("api", "provider", "model", "thinkingLevel")
        if isinstance(message.get(k), str)
    }
    return {KIND: written} if written else None
