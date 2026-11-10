"""Codex, driven through `codex app-server` (JSON-RPC over stdio).

`codex exec --json` cannot be controlled: one process per turn and no input
during a turn. app-server has `turn/start`, `turn/steer` and
`turn/interrupt`.

The protocol shapes were learned by running the server and reading what it
sent (see VERIFIED_VERSION), not from the schema alone.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.metadata
import json
import os
from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING, Any

from ...types import events as events_
from ...types import messages as messages_
from ...types import usage as usage_
from .. import _bridge, _convert, _process, _wire
from .. import _errors as errors
from . import handoff, jsonrpc

if TYPE_CHECKING:
    from .worker import Codex

KIND = "codex"
INSTALL = "npm install -g @openai/codex"
# The CLI version the live tests last passed with. Older ones get a warning.
VERIFIED_VERSION = "0.160.1"

# Server requests that ask for approval. The policy is "never" (workers do
# not ask for approval), so these should not come. If one comes, the answer
# follows the sandbox: accept.
APPROVAL_METHODS = (
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/permissions/requestApproval",
    "execCommandApproval",
    "applyPatchApproval",
)

# Item types that are the agent doing something, not speaking.
TOOL_ITEMS = (
    "fileChange",
    "commandExecution",
    "mcpToolCall",
    "webSearch",
    "toolCall",
)

AUTH_HINT = (
    "codex uses your own login: run `codex login`, or pass "
    "env={'OPENAI_API_KEY': ...}, or point it at another provider with "
    "config={'model_provider': ..., 'model_providers.<name>.base_url': ...}."
)
# The CLI's own phrases for "no usable credentials".
AUTH_MARKERS = ("please run `codex login`",)


class CodexWire:
    def __init__(self, worker: Codex) -> None:
        self.worker = worker
        # Codex records a thread's directory as its real path (on macOS a
        # temp dir /var/... is recorded as /private/var/..., measured).
        self.cwd = os.path.realpath(worker.cwd)
        self.process: _process.Process | None = None
        self.peer: jsonrpc.JsonRpcPeer | None = None
        self.thread = ""
        self.instructions: str | None = None
        # Unbounded: the reader fills it inline and must not wait.
        self.queue: asyncio.Queue[tuple[str, Any, Any]] = asyncio.Queue()
        # The turn that runs, from the turn/start response. `send` steers it.
        self.running: str | None = None
        self.interrupted = False

    def _rpc(self) -> jsonrpc.JsonRpcPeer:
        if self.peer is None:
            raise errors.WorkerError(f"{KIND}: app-server is not running")
        return self.peer

    async def start(
        self,
        *,
        session_id: str | None,
        bridge_url: str,
        instructions: str | None,
        seed: list[messages_.Message],
    ) -> str:
        worker = self.worker
        executable = worker.provider.executable or "codex"
        self.instructions = instructions
        await _process.probe(
            KIND,
            executable,
            install=INSTALL,
            verified=VERIFIED_VERSION,
            cwd=self.cwd,
            env=worker.provider.env,
        )
        config = {
            f"mcp_servers.{_bridge.SERVER_NAME}.url": bridge_url,
            # An `ai` tool can take longer than Codex's default (60 seconds).
            f"mcp_servers.{_bridge.SERVER_NAME}.tool_timeout_sec": 24 * 3600,
            **worker.config,
        }
        self.process = await _process.spawn(
            [executable, "app-server", *config_args(config)],
            cwd=self.cwd,
            env=worker.provider.env,
        )
        self.peer = jsonrpc.JsonRpcPeer(self.process, kind=KIND)
        self.peer.start(
            self._on_notify, self._on_request, on_crash=self._on_crash
        )
        try:
            await self.peer.call(
                "initialize",
                {
                    "clientInfo": {
                        "name": "ai-python",
                        "title": "AI SDK for Python",
                        "version": importlib.metadata.version("ai"),
                    }
                },
                timeout=60,
            )
            await self.peer.notify("initialized")
            return await self._open(session_id, seed)
        except BaseException:
            # A failed start must not leave the server running.
            await self.close()
            raise

    async def _open(
        self, session_id: str | None, seed: list[messages_.Message]
    ) -> str:
        params = {
            "sandbox": self.worker.sandbox or "danger-full-access",
            # Workers do not ask for approval. A request that nobody answers
            # would freeze the agent (measured).
            "approvalPolicy": "never",
            **(
                {"model": self.worker.model_name}
                if self.worker.model_name
                else {}
            ),
            **(
                {"developerInstructions": self.instructions}
                if self.instructions
                else {}
            ),
        }
        if session_id is not None:
            await self._require_here(session_id)
            try:
                result = await self._rpc().call(
                    "thread/resume", {"threadId": session_id, **params}
                )
            except jsonrpc.JsonRpcError as exc:
                raise errors.WorkerError(
                    f"{KIND}: cannot open thread {session_id}: {exc}"
                ) from exc
            self.thread = str(
                (result or {}).get("thread", {}).get("id") or session_id
            )
            return self.thread
        result = await self._rpc().call(
            "thread/start", {"cwd": self.cwd, **params}
        )
        self.thread = str((result or {}).get("thread", {}).get("id") or "")
        if not self.thread:
            raise errors.WorkerError(
                f"{KIND}: thread/start gave no id: {result}"
            )
        if seed:
            # The thread's model-visible history, before the first turn.
            _convert.warn_unsupported_files(
                seed, KIND, accepts=("image/",), takes="images"
            )
            await self._rpc().call(
                "thread/inject_items",
                {
                    "threadId": self.thread,
                    "items": handoff.to_codex_items(seed),
                },
            )
        return self.thread

    async def _require_here(self, session_id: str) -> None:
        """Refuse a thread that belongs to another directory.

        Codex's thread store is the whole machine's. A thread from another
        project resumes natively, and the agent then talks about files that
        are not here.
        """
        try:
            result = await self._rpc().call(
                "thread/read", {"threadId": session_id}
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.WorkerError(
                f"{KIND}: cannot read thread {session_id}: {exc}"
            ) from exc
        thread = (result or {}).get("thread") or {}
        if not thread:
            raise errors.WorkerError(f"{KIND}: no thread {session_id}")
        cwd = thread.get("cwd")
        if cwd is not None and cwd != self.cwd:
            raise errors.WorkerError(
                f"{KIND}: thread {session_id} belongs to {cwd}, not {self.cwd}"
            )

    async def send(self, message: messages_.Message) -> None:
        _convert.check_parts(message, KIND, ("image/",))
        items = await _input_items(message)
        turn = self.running
        if turn is not None:
            try:
                await self._rpc().call(
                    "turn/steer",
                    {
                        "threadId": self.thread,
                        # A precondition: to steer the wrong turn is worse
                        # than to fail.
                        "expectedTurnId": turn,
                        "input": items,
                    },
                )
            except jsonrpc.JsonRpcError as exc:
                # The turn ended before the steer. Only this explicit
                # rejection proves that Codex did not take the input.
                if exc.code != -32600 or str(exc) not in (
                    "no active turn",
                    "no active turn to steer",
                ):
                    raise errors.WorkerError(
                        f"{KIND}: steer rejected: {exc}"
                    ) from exc
            else:
                self.queue.put_nowait(("accepted", message, turn))
                return
        try:
            begun = await self._rpc().call(
                "turn/start", {"threadId": self.thread, "input": items}
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.turn_failed(
                KIND, str(exc), AUTH_HINT, AUTH_MARKERS
            ) from exc
        # The turn id comes from the RESPONSE: this server sends no
        # `turn/started`, and steer and interrupt need the id.
        turn = ((begun or {}).get("turn") or {}).get("id")
        if not isinstance(turn, str) or not turn:
            raise errors.WorkerError(f"{KIND}: turn/start gave no id: {begun}")
        self.running = turn
        self.interrupted = False
        self.queue.put_nowait(("accepted", message, turn))

    async def interrupt(self) -> None:
        turn = self.running
        if turn is None or self.peer is None:
            return
        self.interrupted = True
        with contextlib.suppress(Exception):
            await self.peer.call(
                "turn/interrupt",
                {"threadId": self.thread, "turnId": turn},
                timeout=30,
            )

    async def close(self) -> None:
        if self.peer is not None:
            await self.peer.close()
            self.peer = None
        if self.process is not None:
            with contextlib.suppress(Exception):
                await self.process.terminate()
            self.process = None

    # -- from the server ------------------------------------------------------

    async def _on_notify(self, method: str, params: dict[str, Any]) -> None:
        thread = params.get("threadId") or (params.get("thread") or {}).get(
            "id"
        )
        if thread == self.thread:
            self.queue.put_nowait(("notify", method, params))

    def _on_crash(self, exc: BaseException) -> None:
        self.queue.put_nowait(("crashed", str(exc), None))

    async def _on_request(
        self, request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        # Answer, never leave the agent waiting on a question.
        decision = "accept" if method in APPROVAL_METHODS else "decline"
        await self._rpc().respond(request_id, {"decision": decision})

    async def events(self) -> AsyncGenerator[_wire.Item]:
        current: str | None = None  # the turn whose notifications count
        old: set[str] = set()
        # Notifications of a turn whose turn/start response is not read yet.
        early: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        state = _TurnState()
        while True:
            kind, first, second = await self.queue.get()
            if kind == "crashed":
                raise errors.WorkerError(f"{KIND} crashed: {first}")
            if kind == "accepted":
                yield _wire.Accepted(first)
                if second != current:
                    if current is not None:
                        old.add(current)
                    current = second
                    state = _TurnState()
                    for method, params in early.pop(second, []):
                        for item in self._translate(method, params, state):
                            yield item
                continue
            method, params = first, second
            turn = params.get("turnId") or (params.get("turn") or {}).get("id")
            if turn is not None and turn != current:
                if turn not in old:
                    early.setdefault(turn, []).append((method, params))
                # An interrupted turn's late notifications: without this,
                # the next turn ended on the OLD turn's completion.
                continue
            for item in self._translate(method, params, state):
                yield item

    def _translate(
        self, method: str, params: dict[str, Any], state: _TurnState
    ) -> Iterator[_wire.Item]:
        if method == "thread/tokenUsage/updated":
            state.usage = _usage(params.get("tokenUsage") or {})
        elif method == "item/agentMessage/delta":
            item_id = params.get("itemId", "")
            if item_id not in state.text:
                state.text.add(item_id)
                yield events_.TextStart(block_id=item_id)
            yield events_.TextDelta(
                chunk=params.get("delta", ""), block_id=item_id
            )
        elif method in (
            "item/reasoning/textDelta",
            "item/reasoning/summaryTextDelta",
        ):
            item_id = params.get("itemId", "")
            if item_id not in state.reasoning:
                state.reasoning.add(item_id)
                yield events_.ReasoningStart(block_id=item_id)
            yield events_.ReasoningDelta(
                chunk=params.get("delta", ""), block_id=item_id
            )
        elif method == "item/started":
            item = params.get("item") or {}
            if item.get("type") in TOOL_ITEMS and not _is_bridge(item):
                state.tools.add(item.get("id", ""))
                yield events_.BuiltinToolStart(
                    tool_call_id=item.get("id", ""), tool_name=_tool_name(item)
                )
        elif method == "item/completed":
            yield from _completed(params.get("item") or {}, state)
        elif method == "turn/completed":
            completed = params.get("turn") or {}
            status = completed.get("status")
            if self.running == completed.get("id"):
                self.running = None
            if status == "failed" and not self.interrupted:
                # Reported as a normal stop, "401 Unauthorized" was silence.
                error = errors.turn_failed(
                    KIND,
                    _error_message(completed) or state.last_error,
                    AUTH_HINT,
                    AUTH_MARKERS,
                )
                yield _wire.TurnEnd(
                    reason="error", usage=state.usage, error=error
                )
            elif status == "interrupted" or self.interrupted:
                yield _wire.TurnEnd(reason="interrupted", usage=state.usage)
            else:
                yield _wire.TurnEnd(reason="stop", usage=state.usage)
        elif method == "turn/failed":
            self.running = None
            error = errors.turn_failed(
                KIND, _error_message(params), AUTH_HINT, AUTH_MARKERS
            )
            yield _wire.TurnEnd(reason="error", usage=state.usage, error=error)
        elif method == "error":
            state.last_error = _error_message(params) or state.last_error


class _TurnState:
    """What one turn's translation remembers."""

    def __init__(self) -> None:
        self.text: set[str] = set()
        self.reasoning: set[str] = set()
        self.tools: set[str] = set()
        self.usage: usage_.Usage | None = None
        # The last transport error codex reported while it retried: if the
        # turn fails without a message of its own, the only explanation.
        self.last_error = ""


def _completed(item: dict[str, Any], state: _TurnState) -> Iterator[_wire.Item]:
    item_id = item.get("id", "")
    kind = item.get("type")
    if kind == "agentMessage":
        if item_id in state.text:
            state.text.discard(item_id)
            yield events_.TextEnd(block_id=item_id)
        elif item.get("text"):
            # A message that was not streamed.
            yield events_.TextStart(block_id=item_id)
            yield events_.TextDelta(chunk=item["text"], block_id=item_id)
            yield events_.TextEnd(block_id=item_id)
    elif kind == "reasoning":
        if item_id in state.reasoning:
            state.reasoning.discard(item_id)
            yield events_.ReasoningEnd(block_id=item_id)
        elif text := _reasoning_text(item):
            yield events_.ReasoningStart(block_id=item_id)
            yield events_.ReasoningDelta(chunk=text, block_id=item_id)
            yield events_.ReasoningEnd(block_id=item_id)
    elif kind in TOOL_ITEMS and not _is_bridge(item):
        name = _tool_name(item)
        yield from _convert.builtin_call(
            item_id, name, json.dumps(_tool_args(item)), started=state.tools
        )
        yield _convert.builtin_result(
            item_id,
            name,
            _item_result(item),
            is_error=item.get("status") == "failed",
        )


def _is_bridge(item: dict[str, Any]) -> bool:
    """Tell if `item` calls the MCP bridge (the worker reports those)."""
    return (
        item.get("type") == "mcpToolCall"
        and item.get("server") == _bridge.SERVER_NAME
    )


async def _input_items(message: messages_.Message) -> list[dict[str, Any]]:
    """Return a user message as app-server `UserInput` items.

    An image goes as a data URL, built here from its bytes.
    """
    items: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, messages_.TextPart):
            items.append({"type": "text", "text": part.text})
        elif isinstance(part, messages_.FilePart):
            if part.media_type.startswith("text/"):
                items.append(
                    {"type": "text", "text": await _convert.as_text(part)}
                )
            else:
                encoded = base64.b64encode(await _convert.data(part)).decode(
                    "ascii"
                )
                items.append(
                    {
                        "type": "image",
                        "url": f"data:{part.media_type};base64,{encoded}",
                    }
                )
    return items


def _tool_name(item: dict[str, Any]) -> str:
    # Never the command line: that made every shell call a tool with
    # another name. The command is in the arguments.
    return (
        item.get("toolName") or item.get("server") or item.get("type") or "tool"
    )


def _tool_args(item: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in item.items()
        if k in ("command", "changes", "arguments", "query", "path", "cwd")
    }


def _item_result(item: dict[str, Any]) -> Any:
    """Return what a tool item produced, under the name codex uses.

    A command's output is `aggregatedOutput` (codex 0.160; with only `output`
    every command result was None), a file change's `changes`, an MCP call's
    `result`.
    """
    for key in ("aggregatedOutput", "output", "changes", "result"):
        if item.get(key) is not None:
            return item[key]
    return None


def _reasoning_text(item: dict[str, Any]) -> str:
    """Codex reports reasoning as text, or as a LIST of summary parts."""
    text = item.get("text")
    if isinstance(text, str) and text:
        return text
    summary = item.get("summary")
    if isinstance(summary, str):
        return summary
    if isinstance(summary, list):
        return "".join(part for part in summary if isinstance(part, str))
    return ""


def _usage(raw: dict[str, Any]) -> usage_.Usage:
    totals = raw.get("lastTurn") or raw.get("total") or raw

    def count(*names: str) -> int:
        for name in names:
            value = totals.get(name)
            if isinstance(value, int):
                return value
        return 0

    return usage_.Usage(
        input_tokens=count("inputTokens", "input_tokens"),
        output_tokens=count("outputTokens", "output_tokens"),
        cache_read_tokens=count("cachedInputTokens", "cached_input_tokens"),
        reasoning_tokens=count(
            "reasoningOutputTokens", "reasoning_output_tokens"
        ),
        raw={"codex": raw},
    )


def _error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        detail = error.get("additionalDetails") or ""
        message = str(error.get("message") or "")
        return f"{message} {detail}".strip()
    return str(error or "")


def config_args(config: dict[str, Any]) -> list[str]:
    """`-c key=value` pairs, the values as TOML.

    JSON string, number and bool literals are valid TOML.
    """
    args: list[str] = []
    for key, value in config.items():
        args += ["-c", f"{key}={json.dumps(value)}"]
    return args
