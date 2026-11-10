"""Claude Code, driven through claude-agent-sdk.

The session id we give the CLI is its own session id: `claude --resume <id>`
opens the same conversation, here or in the CLI's own UI.

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
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from ...providers import _optional
from ...types import events as events_
from ...types import messages as messages_
from ...types import usage as usage_
from .. import _bridge, _convert, _process, _wire
from .. import _errors as errors
from . import handoff

if TYPE_CHECKING:
    from .worker import ClaudeCode

KIND = "claude-code"
INSTALL = "npm install -g @anthropic-ai/claude-code"
# The CLI version the live tests last passed with. Older ones get a warning.
VERIFIED_VERSION = "2.1.319"
# Claude Code names a bridge tool `mcp__ai_sdk__<name>`.
BRIDGE_PREFIX = f"mcp__{_bridge.SERVER_NAME}__"

AUTH_HINT = (
    "claude-code uses your own login: run `claude` once and sign in, or "
    "pass env={'ANTHROPIC_API_KEY': ...}."
)
# The CLI's own phrases for "no usable credentials".
AUTH_MARKERS = ("please run /login",)


class ClaudeCodeWire:
    def __init__(self, worker: ClaudeCode) -> None:
        self.worker = worker
        # The CLI keys its store by the real path of its directory.
        self.cwd = os.path.realpath(worker.cwd)
        self.executable = worker.provider.executable or "claude"
        self.client: Any = None
        # Messages written to the CLI and not yet echoed, by the uuid we gave
        # them. See `send()`.
        self.pending: dict[str, messages_.Message] = {}
        # We asked for an interrupt: the next result is not a failure.
        self.interrupted = False

    def _client(self) -> Any:
        if self.client is None:
            raise errors.WorkerError(f"{KIND}: the CLI is not running")
        return self.client

    async def start(
        self,
        *,
        bridge_url: str,
        instructions: str | None,
        seed: list[messages_.Message],
    ) -> str:
        worker = self.worker
        await _process.probe(
            KIND,
            self.executable,
            install=INSTALL,
            verified=VERIFIED_VERSION,
            cwd=self.cwd,
            env=worker.provider.env,
        )
        claude_agent_sdk = _sdk()
        session_id = worker.session_id
        if session_id is not None:
            await self._require_known(session_id)
            resume: str | None = session_id
        elif seed:
            session_id = resume = await self._stage(seed)
        else:
            session_id, resume = str(uuid.uuid4()), None
        options: dict[str, Any] = {
            "cwd": self.cwd,
            "cli_path": self.executable,
            # Raw Anthropic stream events, which map 1:1 onto Start/Delta/End
            # triples. Without them the SDK yields only whole messages.
            "include_partial_messages": True,
            "session_id": session_id if resume is None else None,
            "resume": resume,
            # The CLI echoes each user message from stdin when it takes it
            # into the conversation; `send()` uses that as the acceptance.
            "extra_args": {"replay-user-messages": None},
            "mcp_servers": {
                _bridge.SERVER_NAME: {"type": "http", "url": bridge_url}
            },
            # Workers do not ask for approval. No gate at all: an always-allow
            # callback froze the agent at its next tool call once nobody
            # answered (measured). The allow rule keeps the bridge tools
            # usable even if the user's settings narrow the mode.
            "permission_mode": "bypassPermissions",
            "allowed_tools": [f"mcp__{_bridge.SERVER_NAME}"],
        }
        # claude-agent-sdk spawns the CLI itself: pass the env along. An
        # `ai` tool can take longer than Claude Code's default MCP tool
        # timeout; the user's env wins.
        options["env"] = {
            "MCP_TOOL_TIMEOUT": str(24 * 3600 * 1000),
            **worker.provider.env,
        }
        if worker.model_name:
            options["model"] = worker.model_name
        if instructions:
            # Added to Claude Code's own system prompt, never replacing it.
            options["system_prompt"] = {
                "type": "preset",
                "preset": "claude_code",
                "append": instructions,
            }
        if worker.allowed_tools is not None:
            options["tools"] = worker.allowed_tools
        if worker.disallowed_tools:
            options["disallowed_tools"] = worker.disallowed_tools
        if worker.max_turns is not None:
            options["max_turns"] = worker.max_turns
        # Set before connect: a failed connect must still be closed.
        self.client = claude_agent_sdk.ClaudeSDKClient(
            options=claude_agent_sdk.ClaudeAgentOptions(**options)
        )
        try:
            await self.client.connect()
        except claude_agent_sdk.ClaudeSDKError as exc:
            raise _sdk_error(exc) from exc
        return session_id

    async def _stage(self, seed: list[messages_.Message]) -> str:
        """Write a past into the CLI's store under a new id; run nothing."""
        claude_agent_sdk = _sdk()
        session_id = str(uuid.uuid4())
        _convert.warn_unsupported_files(
            seed,
            KIND,
            accepts=("image/", "application/pdf"),
            takes="images and PDFs",
        )
        dropped = _convert.reasoning_count(seed)
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
            self.worker.provider.env.get("CLAUDE_CONFIG_DIR")
            or os.environ.get("CLAUDE_CONFIG_DIR")
            or str(pathlib.Path.home() / ".claude")
        )
        key = claude_agent_sdk.project_key_for_directory(self.cwd)
        path = pathlib.Path(config) / "projects" / key / f"{session_id}.jsonl"
        records = await handoff.to_claude_records(seed, session_id, self.cwd)
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
                claude_agent_sdk.get_session_info, session_id, self.cwd
            )
        except Exception as exc:
            raise errors.WorkerError(
                f"{KIND}: cannot read session {session_id}: {exc}"
            ) from exc
        if info is None:
            raise errors.WorkerError(
                f"{KIND}: no session {session_id} in {self.cwd}"
            )

    async def send(self, message: messages_.Message) -> None:
        """Write a user message to the CLI's stdin: a new turn or a steer.

        Acceptance is the CLI's own echo (`--replay-user-messages`), matched
        by a uuid we set: the CLI keeps it (measured). The echo comes when
        the CLI takes the message into the conversation, before the content
        it causes. A message written during a turn is injected into that
        turn at the next step, and the echo comes there, not at the write
        (measured on 2.1.319: written during a Bash call, echoed after its
        result; one result for both messages). An `Accepted` emitted at the
        write would split the reply at the wrong place.
        """
        claude_agent_sdk = _sdk()
        _convert.check_parts(message, KIND, ("image/", "application/pdf"))
        content = await _content(message)
        client = self._client()
        uid = str(uuid.uuid4())
        # Before the write: the echo can come before `query()` returns.
        self.pending[uid] = message
        try:
            await client.query(_one_message(content, uid))
        except BaseException as exc:
            self.pending.pop(uid, None)
            if isinstance(exc, claude_agent_sdk.ClaudeSDKError):
                raise _sdk_error(exc) from exc
            raise

    async def interrupt(self) -> None:
        # The CLI answers with an error result (`error_during_execution`),
        # which `events()` reads as the interrupted turn's end.
        if self.client is None:
            return
        self.interrupted = True
        with contextlib.suppress(Exception):
            await self.client.interrupt()

    async def close(self) -> None:
        client, self.client = self.client, None
        self.pending.clear()
        if client is not None:
            with contextlib.suppress(Exception):
                await client.disconnect()

    # -- from the CLI ---------------------------------------------------------

    async def events(self) -> AsyncGenerator[_wire.Item]:
        claude_agent_sdk = _sdk()
        client = self._client()
        turn = _Turn()
        try:
            async for message in client.receive_messages():
                for item in self._translate(message, turn):
                    yield item
                if isinstance(message, claude_agent_sdk.ResultMessage):
                    turn = _Turn()
        except claude_agent_sdk.ClaudeSDKError as exc:
            raise _sdk_error(exc) from exc
        raise errors.WorkerError(f"{KIND}: the CLI exited")

    def _translate(self, message: Any, turn: _Turn) -> Iterator[_wire.Item]:
        claude_agent_sdk = _sdk()
        if getattr(message, "parent_tool_use_id", None) is not None:
            # A subagent's own steps (inside a Task call). The Task call and
            # its result come at the top level; its inner steps do not go
            # into this conversation.
            return
        if isinstance(message, claude_agent_sdk.StreamEvent):
            yield from turn.stream_event(message)
        elif isinstance(message, claude_agent_sdk.AssistantMessage):
            if message.error:
                turn.error = str(message.error)
            streamed = message.message_id in turn.streamed
            for index, block in enumerate(message.content):
                if isinstance(block, claude_agent_sdk.ToolUseBlock):
                    if block.name.startswith(BRIDGE_PREFIX):
                        continue  # an `ai` tool: the bridge reports it
                    turn.names[block.id] = block.name
                    yield from _convert.builtin_call(
                        block.id,
                        block.name,
                        json.dumps(block.input),
                        started=turn.started,
                    )
                elif (
                    isinstance(block, claude_agent_sdk.TextBlock)
                    and not streamed
                    and block.text
                ):
                    # Not streamed: the CLI made it up, for example
                    # "Invalid API key" (measured, model "<synthetic>").
                    block_id = f"{message.message_id or message.uuid}:{index}"
                    yield events_.TextStart(block_id=block_id)
                    yield events_.TextDelta(chunk=block.text, block_id=block_id)
                    yield events_.TextEnd(block_id=block_id)
        elif isinstance(message, claude_agent_sdk.UserMessage):
            accepted = self.pending.pop(message.uuid or "", None)
            if accepted is not None:
                yield _wire.Accepted(accepted)
                return
            # Tool results, or a user record the CLI wrote itself (for
            # example "[Request interrupted by user for tool use]").
            blocks = (
                message.content if isinstance(message.content, list) else []
            )
            for block in blocks:
                if (
                    isinstance(block, claude_agent_sdk.ToolResultBlock)
                    and block.tool_use_id in turn.names
                ):
                    yield _convert.builtin_result(
                        block.tool_use_id,
                        turn.names[block.tool_use_id],
                        block.content,
                        is_error=bool(block.is_error),
                    )
        elif isinstance(message, claude_agent_sdk.ResultMessage):
            yield from turn.close_blocks()
            usage = _usage(message.usage or {}, message.total_cost_usd)
            interrupted, self.interrupted = self.interrupted, False
            if interrupted:
                # We asked for this. The CLI reports an error result after
                # an interrupt; it is not a failure.
                yield _wire.TurnEnd(reason="interrupted", usage=usage)
            elif message.subtype == "error_max_turns":
                yield _wire.TurnEnd(reason="length", usage=usage)
            elif message.is_error:
                detail = (
                    message.result
                    or "; ".join(message.errors or [])
                    or turn.error
                    or message.subtype
                )
                error = (
                    errors.NotAuthenticatedError(KIND, detail, AUTH_HINT)
                    if turn.error == "authentication_failed"
                    else errors.turn_failed(
                        KIND, detail, AUTH_HINT, AUTH_MARKERS
                    )
                )
                yield _wire.TurnEnd(reason="error", usage=usage, error=error)
            else:
                yield _wire.TurnEnd(reason="stop", usage=usage)


class _Turn:
    """What one turn's translation remembers."""

    def __init__(self) -> None:
        # The API message that streams now, and all that streamed.
        self.message = ""
        self.streamed: set[str] = set()
        # Open text and thinking blocks: content-block index -> (kind, id).
        self.blocks: dict[int, tuple[str, str]] = {}
        # The CLI's own tool calls: started ones, and names by id. Bridge
        # calls are in neither, so their results are skipped too.
        self.started: set[str] = set()
        self.names: dict[str, str] = {}
        # The error an assistant message reported, e.g. authentication_failed.
        self.error = ""

    def stream_event(self, message: Any) -> Iterator[_wire.Content]:
        """One raw Anthropic stream event -> text and reasoning events.

        Block indexes restart at 0 in each API message, and one assistant
        message of the worker can hold several: the block id carries the
        API message id.
        """
        raw: dict[str, Any] = message.event
        kind = raw.get("type")
        index = raw.get("index", 0)
        if kind == "message_start":
            self.message = str(
                (raw.get("message") or {}).get("id") or message.uuid
            )
            self.streamed.add(self.message)
            return
        block_id = f"{self.message}:{index}"
        if kind == "content_block_start":
            block = raw.get("content_block") or {}
            if block.get("type") == "text":
                self.blocks[index] = ("text", block_id)
                yield events_.TextStart(block_id=block_id)
            elif block.get("type") == "thinking":
                self.blocks[index] = ("thinking", block_id)
                yield events_.ReasoningStart(block_id=block_id)
            elif block.get("type") == "tool_use":
                name = str(block.get("name") or "")
                call_id = str(block.get("id") or "")
                if not name.startswith(BRIDGE_PREFIX):
                    self.started.add(call_id)
                    yield events_.BuiltinToolStart(
                        tool_call_id=call_id, tool_name=name
                    )
        elif kind == "content_block_delta":
            delta = raw.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield events_.TextDelta(chunk=delta["text"], block_id=block_id)
            elif delta.get("type") == "thinking_delta" and delta.get(
                "thinking"
            ):
                yield events_.ReasoningDelta(
                    chunk=delta["thinking"], block_id=block_id
                )
        elif kind == "content_block_stop" and index in self.blocks:
            yield from self._end(self.blocks.pop(index))

    def close_blocks(self) -> Iterator[_wire.Content]:
        """End the blocks that an interrupt or a failure cut."""
        blocks, self.blocks = self.blocks, {}
        for block in blocks.values():
            yield from self._end(block)

    @staticmethod
    def _end(block: tuple[str, str]) -> Iterator[_wire.Content]:
        kind, block_id = block
        if kind == "text":
            yield events_.TextEnd(block_id=block_id)
        else:
            yield events_.ReasoningEnd(block_id=block_id)


def _sdk() -> Any:
    return _optional.import_optional_sdk(
        "claude_agent_sdk", provider="claude-code", extra="claude-code"
    )


async def _content(message: messages_.Message) -> list[dict[str, Any]]:
    """Return a user message as Anthropic content blocks.

    Images and PDFs travel as base64 blocks (measured: both read correctly).
    """
    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, messages_.TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, messages_.FilePart):
            if part.media_type.startswith("text/"):
                blocks.append(
                    {"type": "text", "text": await _convert.as_text(part)}
                )
                continue
            kind = (
                "image" if part.media_type.startswith("image/") else "document"
            )
            encoded = base64.b64encode(await _convert.data(part)).decode()
            blocks.append(
                {
                    "type": kind,
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": encoded,
                    },
                }
            )
    return blocks


async def _one_message(
    content: list[dict[str, Any]], uid: str
) -> AsyncIterator[dict[str, Any]]:
    """Yield one stream-json user message, with the uuid the echo carries."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
        "session_id": "default",
        "uuid": uid,
    }


def _usage(raw: dict[str, Any], cost: float | None) -> usage_.Usage:
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    details = raw.get("output_tokens_details") or {}
    thinking = details.get("thinking_tokens")
    return usage_.Usage(
        # As the Anthropic provider: input includes the cache reads.
        input_tokens=int(raw.get("input_tokens") or 0) + cache_read,
        output_tokens=int(raw.get("output_tokens") or 0),
        reasoning_tokens=thinking if isinstance(thinking, int) else None,
        cache_read_tokens=cache_read,
        cache_write_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        # Cost has no home in Usage; the CLI's own number rides in `raw`.
        raw={"cost_usd": cost, "claude": raw},
    )


def _sdk_error(exc: BaseException) -> BaseException:
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
    return errors.WorkerError(f"{KIND} failed: {exc}")
