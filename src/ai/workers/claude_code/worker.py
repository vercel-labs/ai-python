"""Claude Code, driven through claude-agent-sdk.

The session id we hand the CLI IS its own session id, so `claude --resume
<id>` opens the same conversation, here or in the CLI's own UI.

The comments record what was MEASURED against a real `claude` CLI (see
VERIFIED_VERSION), not what the API suggested.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import pathlib
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

import pydantic

from ...providers import _optional
from ...types import events as events_
from ...types import messages as messages_
from ...types import usage as usage_
from .. import _adapter, _bridge, _convert, _handoff, _worker
from .. import _errors as errors
from . import handoff

if TYPE_CHECKING:
    import claude_agent_sdk

KIND = "claude-code"
INSTALL = "npm install -g @anthropic-ai/claude-code"
# The CLI the live tests last passed against. Older ones get a warning.
VERIFIED_VERSION = "2.1.289"

# Seconds the CLI waits for an approval. Its own default (60s) would cut a
# person short: the hook would time out and the call would be denied.
APPROVAL_TIMEOUT = 24 * 3600

AUTH_HINT = (
    "claude-code uses your own login: run `claude` once and sign in, or "
    "pass env={'ANTHROPIC_API_KEY': ...}."
)


class ClaudeCode(_worker.Worker):
    """Claude Code, through `claude-agent-sdk`. Needs the `claude-code` extra.

    `allowed_tools` limits Claude Code's own tools (`["Read", "Grep"]`, or
    `[]` for none), and `disallowed_tools` takes some away. `max_turns` caps
    the agent's steps in one turn; a turn that hits it ends with
    `finish_reason="length"`.
    """

    worker_class_id: Literal["claude-code"] = "claude-code"
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    max_turns: int | None = None

    _mcp_servers: dict[str, dict[str, Any]] = pydantic.PrivateAttr(
        default_factory=dict
    )
    _approval_callback: _adapter.Approve | None = pydantic.PrivateAttr(
        default=None
    )
    _instructions: str | None = pydantic.PrivateAttr(default=None)
    _client: Any = pydantic.PrivateAttr(default=None)
    # Requested interrupt, or unread responses after a failed turn. The
    # next query must drain these before it writes any new input.
    _interrupted: bool = pydantic.PrivateAttr(default=False)
    # A reader owns the responses to all queries accepted into this run.
    _running: bool = pydantic.PrivateAttr(default=False)
    # Account one terminal result per query written to streaming input.
    # The SDK supplies no input-id/result-id correlation: this is not proof
    # of native consumption, nor accounting for autonomous/background turns.
    _pending_results: int = pydantic.PrivateAttr(default=0)
    _input_lock: asyncio.Lock = pydantic.PrivateAttr(
        default_factory=asyncio.Lock
    )

    @property
    def _cwd(self) -> str:
        # The CLI keys its store by the real path of its directory.
        return os.path.realpath(self.cwd)

    @property
    def _executable(self) -> str:
        return self.executable or "claude"

    async def _cli_start(self, mcp_servers: dict[str, dict[str, Any]]) -> None:
        _sdk()
        self._mcp_servers = dict(mcp_servers)
        self._approval_callback = self._approve if self.approve else None
        self._cli_version = await _adapter.probe(
            KIND,
            self._executable,
            install=INSTALL,
            verified=VERIFIED_VERSION,
            cwd=self._cwd,
            env=self.env,
        )

    async def _cli_close(self) -> None:
        client, self._client = self._client, None
        self._running = False
        self._pending_results = 0
        self._interrupted = False
        self._instructions = None
        self._approval_callback = None
        self._mcp_servers = {}
        if client is not None:
            with contextlib.suppress(Exception):
                await client.disconnect()

    # -- the conversation -----------------------------------------------------

    async def _cli_open(
        self,
        *,
        session_id: str | None,
        fork: bool,
        seed: list[messages_.Message] | None,
        instructions: str | None,
    ) -> str:
        claude_agent_sdk = _sdk()
        self._instructions = instructions
        if seed:
            session_id = await self._stage(seed)
        elif session_id is not None:
            await self._require_known(session_id)
            if fork:
                try:
                    # The SDK remaps every UUID and rewrites the chain.
                    result = await asyncio.to_thread(
                        claude_agent_sdk.fork_session, session_id, self._cwd
                    )
                except Exception as exc:
                    raise errors.WorkerError(
                        f"{KIND}: cannot fork session {session_id}: {exc}"
                    ) from exc
                session_id = result.session_id
        if session_id is None:
            session_id = str(uuid.uuid4())
            await self._connect(session_id, resume=None)
        else:
            await self._connect(session_id, resume=session_id)
        return session_id

    async def _stage(self, history: list[messages_.Message]) -> str:
        """Write a past into the CLI's store under a fresh id; run nothing."""
        claude_agent_sdk = _sdk()
        session_id = str(uuid.uuid4())
        _handoff.warn_unsupported_files(history, KIND)
        dropped = _handoff.reasoning_count(history)
        if dropped:
            warnings.warn(
                f"{dropped} reasoning part(s) in the history were not carried "
                "into claude-code: Anthropic's safeguards refuse replayed "
                "model reasoning ([reasoning_extraction]). Text and tool "
                "calls were carried in full.",
                stacklevel=2,
            )
        # Where the CLI keeps transcripts:
        # `$CLAUDE_CONFIG_DIR/projects/<key>/<session>.jsonl`.
        config = (
            self.env.get("CLAUDE_CONFIG_DIR")
            or os.environ.get("CLAUDE_CONFIG_DIR")
            or str(pathlib.Path.home() / ".claude")
        )
        key = claude_agent_sdk.project_key_for_directory(self._cwd)
        path = pathlib.Path(config) / "projects" / key / f"{session_id}.jsonl"
        records = handoff.to_claude_records(history, session_id, self._cwd)
        text = "".join(
            json.dumps(r, separators=(",", ":")) + "\n" for r in records
        )

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        await asyncio.to_thread(write)
        return session_id

    async def _require_known(self, session_id: str) -> None:
        """Refuse an unknown session before connecting to it.

        Connecting to a ghost surfaces as claude-agent-sdk's own error
        somewhere downstream.
        """
        claude_agent_sdk = _sdk()
        try:
            info = await asyncio.to_thread(
                claude_agent_sdk.get_session_info, session_id, self._cwd
            )
        except Exception as exc:
            raise errors.WorkerError(
                f"{KIND}: cannot read session {session_id}: {exc}"
            ) from exc
        if info is None:
            raise errors.WorkerError(
                f"{KIND}: no session {session_id} in {self._cwd}"
            )

    async def _connect(self, session_id: str, *, resume: str | None) -> None:
        claude_agent_sdk = _sdk()
        options: dict[str, Any] = {
            "cwd": self._cwd,
            "cli_path": self._executable,
            # Raw Anthropic stream events, which map 1:1 onto Start/Delta/End
            # triples. Without them the SDK yields only whole messages.
            "include_partial_messages": True,
            "session_id": session_id if resume is None else None,
            "resume": resume,
        }
        if self.env:
            # claude-agent-sdk spawns the CLI itself: pass the env along.
            options["env"] = dict(self.env)
        if self.model_name:
            options["model"] = self.model_name
        if self._instructions:
            # Added to Claude Code's own system prompt, never replacing it.
            options["system_prompt"] = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._instructions,
            }
        if self._mcp_servers:
            options["mcp_servers"] = self._mcp_servers
        if self.allowed_tools is not None:
            options["tools"] = self.allowed_tools
        if self.disallowed_tools:
            options["disallowed_tools"] = self.disallowed_tools
        if self.max_turns is not None:
            options["max_turns"] = self.max_turns
        if self.effort:
            # Reasoning effort, with adaptive thinking so the budget follows.
            options["effort"] = self.effort
            options["thinking"] = {"type": "adaptive"}
        if self._approval_callback is not None:
            # A PreToolUse hook sees EVERY tool call, read-only ones included
            # (measured). The permission callback alone only sees what the
            # CLI's policy asks about; it stays as a safety net.
            options["hooks"] = {
                "PreToolUse": [
                    claude_agent_sdk.HookMatcher(
                        matcher=None,
                        hooks=[self._pre_tool_use],
                        timeout=APPROVAL_TIMEOUT,
                    )
                ]
            }
            options["can_use_tool"] = self._can_use_tool
        else:
            # No gate: an always-allow callback would freeze the agent at
            # its next tool call once nobody answers (measured).
            options["permission_mode"] = "bypassPermissions"
        client = claude_agent_sdk.ClaudeSDKClient(
            options=claude_agent_sdk.ClaudeAgentOptions(**options)
        )
        try:
            await client.connect()
        except claude_agent_sdk.ClaudeSDKError as exc:
            raise _translate(exc) from exc
        self._client = client

    # -- approval -------------------------------------------------------------

    async def _pre_tool_use(
        self, data: Any, tool_use_id: str | None, ctx: Any
    ) -> Any:
        assert self._approval_callback is not None
        granted, reason = await self._approval_callback(
            tool_use_id or str(data.get("tool_use_id") or uuid.uuid4()),
            str(data.get("tool_name") or ""),
            dict(data.get("tool_input") or {}),
        )
        output: dict[str, Any] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if granted else "deny",
        }
        if not granted:
            output["permissionDecisionReason"] = reason
        return {"hookSpecificOutput": output}

    async def _can_use_tool(
        self,
        name: str,
        tool_input: dict[str, Any],
        ctx: claude_agent_sdk.ToolPermissionContext,
    ) -> claude_agent_sdk.PermissionResult:
        import claude_agent_sdk  # noqa: PLC0415

        assert self._approval_callback is not None
        granted, reason = await self._approval_callback(
            str(getattr(ctx, "tool_use_id", None) or uuid.uuid4()),
            name,
            dict(tool_input),
        )
        if granted:
            return claude_agent_sdk.PermissionResultAllow()
        return claude_agent_sdk.PermissionResultDeny(message=reason)

    # -- turns ----------------------------------------------------------------

    async def _cli_turn(
        self, prompt: _adapter.Prompt
    ) -> AsyncGenerator[events_.Event]:
        claude_agent_sdk = _sdk()
        _convert.check_files(prompt, KIND, ("image/", "application/pdf"))
        client = self._client
        assert client is not None
        # Interruption or failure can leave accepted responses unread.
        # Drain them first, or THIS turn could settle on an earlier result.
        await self._drain_interrupted()
        content = await _content(prompt)
        names: dict[str, str] = {}
        open_blocks: dict[int, str] = {}
        try:
            if isinstance(content, str):
                await client.query(content)
            else:
                await client.query(_one_message(content))
        except claude_agent_sdk.ClaudeSDKError as exc:
            raise _translate(exc) from exc
        self._pending_results = 1
        self._running = True
        try:
            # query() completed its write. The base uses this as the send
            # acknowledgement; waiting for output can deadlock approvals.
            yield events_.StreamStart()
            # receive_response() stops at the FIRST result, even if a duplex
            # query has already queued another response. Keep one reader
            # until every accepted query has a terminal result.
            async with contextlib.aclosing(
                _guard(client.receive_messages())
            ) as incoming:
                async for message in incoming:
                    if isinstance(message, claude_agent_sdk.StreamEvent):
                        # Whole AssistantMessages repeat streamed text; only
                        # tool calls are taken from them below.
                        for event in _stream_events(message.event, open_blocks):
                            yield event
                    elif isinstance(message, claude_agent_sdk.AssistantMessage):
                        for block in message.content:
                            if isinstance(block, claude_agent_sdk.ToolUseBlock):
                                names[block.id] = block.name
                                for event in _convert.builtin_call(
                                    block.id,
                                    block.name,
                                    json.dumps(block.input),
                                    started=set(),
                                ):
                                    yield event
                    elif isinstance(message, claude_agent_sdk.UserMessage):
                        content_blocks = (
                            message.content
                            if isinstance(message.content, list)
                            else []
                        )
                        for block in content_blocks:
                            if isinstance(
                                block, claude_agent_sdk.ToolResultBlock
                            ):
                                yield _convert.builtin_result(
                                    block.tool_use_id,
                                    names.get(block.tool_use_id, ""),
                                    block.content,
                                    is_error=bool(block.is_error),
                                )
                    elif isinstance(message, claude_agent_sdk.ResultMessage):
                        # A write that began before this result must finish
                        # accounting before we decide to retire this reader.
                        # Never hold this lock across a yield to the caller.
                        async with self._input_lock:
                            self._pending_results -= 1
                            finished = self._pending_results == 0
                            self._running = not finished
                            interrupted = self._interrupted
                            self._interrupted = False
                        try:
                            reason = _finish_reason(
                                message, interrupted=interrupted
                            )
                        except errors.WorkerError:
                            # stream()/generate() stop at StreamEnd: raising
                            # after it would turn this failure into success.
                            # Stop accepting input now, before closing the SDK
                            # iterator can yield control. Preserve response
                            # debt for the next turn's drain (or close).
                            self._running = False
                            self._interrupted = not finished
                            raise
                        yield events_.StreamEnd(
                            usage=_usage(message), finish_reason=reason
                        )
                        if finished:
                            return
                        names.clear()
                        open_blocks.clear()
                        yield events_.StreamStart()
                raise errors.WorkerError(
                    f"{KIND}: CLI output ended with "
                    f"{self._pending_results} response(s) outstanding"
                )
        finally:
            self._running = False

    async def _drain_interrupted(self) -> None:
        if not self._interrupted or self._client is None:
            return
        claude_agent_sdk = _sdk()
        remaining = max(1, self._pending_results)
        # Cancellation or failure can leave several accepted responses.
        # Drain them all before allowing another query on this connection.
        async with (
            asyncio.timeout(30),
            contextlib.aclosing(
                _guard(self._client.receive_messages())
            ) as incoming,
        ):
            async for message in incoming:
                if isinstance(message, claude_agent_sdk.ResultMessage):
                    remaining -= 1
                    self._pending_results = remaining
                    if remaining == 0:
                        self._interrupted = False
                        return
        raise errors.WorkerError(f"{KIND}: CLI output ended during interrupt")

    async def _cli_send_running(self, prompt: _adapter.Prompt) -> bool:
        if not self._running:
            return False
        _convert.check_files(prompt, KIND, ("image/", "application/pdf"))
        content = await _content(prompt)
        # Reading a file (or waiting for another writer) can yield while the
        # response finishes. If it did, the base can start a fresh iterator.
        async with self._input_lock:
            if not self._running:
                return False
            client = self._client
            if client is None:
                raise errors.WorkerError(f"{KIND}: no CLI connection")
            claude_agent_sdk = _sdk()
            try:
                await client.query(
                    content
                    if isinstance(content, str)
                    else _one_message(content)
                )
            except claude_agent_sdk.ClaudeSDKError as exc:
                raise _translate(exc) from exc
            # The result reader uses this same lock, so it cannot retire
            # between the successful write and recording its response debt.
            # This acknowledges transport acceptance, not native processing.
            self._pending_results += 1
            return True

    async def _cli_interrupt(self) -> None:
        # Only a running turn: the CLI answers an interrupt with a result,
        # which the next turn reads first (see `_drain_interrupted`).
        if self._client is None or not self._running:
            return
        self._interrupted = True
        with contextlib.suppress(Exception):
            await self._client.interrupt()

    # -- the store ------------------------------------------------------------

    async def _cli_history(self, session_id: str) -> list[messages_.Message]:
        """Read the CLI's own transcript, every content block included."""
        claude_agent_sdk = _sdk()
        await self._require_known(session_id)
        stored = await asyncio.to_thread(
            claude_agent_sdk.get_session_messages, session_id, self._cwd
        )
        return _read(stored)

    async def _cli_list_sessions(self) -> list[_adapter.SessionInfo]:
        claude_agent_sdk = _sdk()
        infos = await asyncio.to_thread(
            claude_agent_sdk.list_sessions, self._cwd
        )
        return [
            _adapter.SessionInfo(
                session_id=i.session_id,
                title=i.custom_title or i.summary or i.first_prompt,
                cwd=i.cwd,
                created_at=i.created_at,
                updated_at=i.last_modified,
            )
            for i in infos
        ]

    def _cli_tui_command(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        running_as = session_id or str(uuid.uuid4())
        argv = [self._executable]
        argv += (
            ["--resume", session_id]
            if session_id
            else ["--session-id", running_as]
        )
        if self.model_name:
            argv += ["--model", self.model_name]
        if self._instructions:
            argv += ["--append-system-prompt", self._instructions]
        servers = {
            k: v
            for k, v in self.mcp_servers.items()
            if k != _bridge.SERVER_NAME
        }
        if servers:
            argv += ["--mcp-config", json.dumps({"mcpServers": servers})]
        if self.allowed_tools is not None:
            argv += ["--tools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            argv += ["--disallowedTools", ",".join(self.disallowed_tools)]
        # Measured: the TUI otherwise updates itself in the background and
        # announces "Restart to apply" mid-session.
        env = {"DISABLE_AUTOUPDATER": "1", **self.env}
        return argv, env, running_as


def _sdk() -> Any:
    return _optional.import_optional_sdk(
        "claude_agent_sdk", extra="claude-code", feature="ai.workers.ClaudeCode"
    )


async def _content(prompt: _adapter.Prompt) -> str | list[dict[str, Any]]:
    """Return a prompt as the CLI takes it: a string, or content blocks.

    Images and PDFs travel as base64 blocks (measured: both read correctly).
    """
    if len(prompt) == 1 and isinstance(prompt[0], messages_.TextPart):
        return prompt[0].text
    blocks: list[dict[str, Any]] = []
    for part in prompt:
        if isinstance(part, messages_.TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif part.media_type.startswith("text/"):
            blocks.append(
                {"type": "text", "text": await _convert.as_text(part)}
            )
        else:
            kind = (
                "image" if part.media_type.startswith("image/") else "document"
            )
            blocks.append(
                {
                    "type": kind,
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": base64.b64encode(
                            await _convert.data(part)
                        ).decode("ascii"),
                    },
                }
            )
    return blocks


async def _one_message(
    content: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Yield one stream-json user message, the way `query()` sends a str."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
        "session_id": "default",
    }


def _usage(message: Any) -> usage_.Usage:
    raw = message.usage or {}
    return usage_.Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cache_read_tokens=int(raw.get("cache_read_input_tokens", 0)),
        cache_write_tokens=int(raw.get("cache_creation_input_tokens", 0)),
        # Cost has no home in Usage; the CLI's own number rides in `raw`.
        raw={"cost_usd": message.total_cost_usd, "claude": raw},
    )


def _finish_reason(message: Any, *, interrupted: bool) -> str:
    subtype = message.subtype or ""
    if interrupted:
        # We asked for this. The CLI reports an error result after an
        # interrupt; it is not a failure.
        return "cancelled"
    if subtype == "error_max_turns":
        return "length"
    if getattr(message, "is_error", False):
        detail = str(getattr(message, "result", None) or subtype)
        raise errors.turn_failed(KIND, detail, AUTH_HINT)
    return "stop"


def _translate(exc: BaseException) -> BaseException:
    """Turn a claude-agent-sdk exception into a worker error."""
    claude_agent_sdk = _sdk()
    if isinstance(exc, claude_agent_sdk.CLINotFoundError):
        return errors.ExecutableMissingError(KIND, "claude", INSTALL)
    if isinstance(exc, claude_agent_sdk.ProcessError):
        code = getattr(exc, "exit_code", None)
        return errors.WorkerError(f"{KIND} crashed (exit {code}): {exc}")
    if isinstance(exc, claude_agent_sdk.CLIConnectionError):
        return errors.WorkerError(f"{KIND}: lost the CLI connection: {exc}")
    if isinstance(exc, claude_agent_sdk.CLIJSONDecodeError):
        return errors.WorkerError(f"{KIND}: unparseable CLI output: {exc}")
    return exc


async def _guard(stream: Any) -> Any:
    """Re-raise claude-agent-sdk failures as worker errors, mid-stream too."""
    claude_agent_sdk = _sdk()
    try:
        async with contextlib.aclosing(stream):
            async for item in stream:
                yield item
    except claude_agent_sdk.ClaudeSDKError as exc:
        raise _translate(exc) from exc


def _stream_events(
    raw: dict[str, Any], open_blocks: dict[int, str]
) -> list[events_.Event]:
    """One raw Anthropic stream event -> text and reasoning events.

    `open_blocks` tracks which content-block index is text or thinking, so
    the closing event emits the matching End.
    """
    kind = raw.get("type")
    index = raw.get("index", 0)
    out: list[events_.Event] = []
    block_id = f"b{index}"
    if kind == "content_block_start":
        block_type = (raw.get("content_block") or {}).get("type")
        if block_type == "text":
            open_blocks[index] = "text"
            out.append(events_.TextStart(block_id=block_id))
        elif block_type == "thinking":
            open_blocks[index] = "thinking"
            out.append(events_.ReasoningStart(block_id=block_id))
    elif kind == "content_block_delta":
        delta = raw.get("delta") or {}
        if delta.get("type") == "text_delta" and delta.get("text"):
            out.append(
                events_.TextDelta(chunk=delta["text"], block_id=block_id)
            )
        elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
            out.append(
                events_.ReasoningDelta(
                    chunk=delta["thinking"], block_id=block_id
                )
            )
    elif kind == "content_block_stop":
        opened = open_blocks.pop(index, None)
        if opened == "text":
            out.append(events_.TextEnd(block_id=block_id))
        elif opened == "thinking":
            out.append(events_.ReasoningEnd(block_id=block_id))
    return out


def _read(records: list[Any]) -> list[messages_.Message]:
    """Read stored transcript records as messages, like a live turn's.

    The CLI's tools are built-in parts. On disk a tool result is a block in
    the NEXT user record; here it joins the assistant message of its call.
    """
    out: list[messages_.Message] = []
    calls: dict[str, tuple[messages_.Message, messages_.BuiltinToolCallPart]]
    calls = {}
    for record in records:
        role, parts, results = _record(record)
        if parts and role is not None:
            message = messages_.Message(role=role, parts=parts)
            out.append(message)
            for part in parts:
                if isinstance(part, messages_.BuiltinToolCallPart):
                    calls[part.tool_call_id] = (message, part)
        for block in results:
            found = calls.get(block.get("tool_use_id", ""))
            if found is None:
                continue  # a result without its call: nothing to join
            message, call = found
            message.parts.append(
                messages_.BuiltinToolReturnPart(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    result=block.get("content"),
                    is_error=bool(block.get("is_error")),
                )
            )
    return out


def _record(
    record: Any,
) -> tuple[
    Literal["user", "assistant"] | None, list[Any], list[dict[str, Any]]
]:
    """Return a record's role, its parts, and its tool result blocks.

    The role is None for a record that is neither user nor assistant.
    """
    raw = record.message if isinstance(record.message, dict) else {}
    role = raw.get("role") or record.type
    content = raw.get("content")
    if isinstance(content, str):
        blocks: list[Any] = [{"type": "text", "text": content}]
    else:
        blocks = content if isinstance(content, list) else []
    parts: list[Any] = []
    results: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            parts.append(messages_.TextPart(text=block["text"]))
        elif kind in ("image", "document"):
            found = block.get("source")
            source: dict[str, Any] = found if isinstance(found, dict) else {}
            if source.get("type") == "base64" and source.get("data"):
                fallback = "image/png" if kind == "image" else "application/pdf"
                parts.append(
                    messages_.FilePart(
                        data=str(source["data"]),
                        media_type=str(source.get("media_type") or fallback),
                    )
                )
            elif source.get("type") == "url" and source.get("url"):
                parts.append(messages_.FilePart.from_url(str(source["url"])))
        elif kind == "thinking" and block.get("thinking"):
            parts.append(messages_.ReasoningPart(text=block["thinking"]))
        elif kind == "tool_use":
            parts.append(
                messages_.BuiltinToolCallPart(
                    tool_call_id=block.get("id", ""),
                    tool_name=block.get("name", ""),
                    tool_args=json.dumps(block.get("input") or {}),
                )
            )
        elif kind == "tool_result":
            results.append(block)
    return (role if role in ("user", "assistant") else None), parts, results
