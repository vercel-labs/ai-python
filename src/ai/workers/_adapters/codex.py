"""Codex, driven through `codex app-server`.

Chosen over `codex exec --json`, which cannot be controlled: one process per
turn, stdin unused, no approval channel. app-server has `turn/steer`,
`turn/interrupt` and `item/*/requestApproval`.

The protocol shapes here were learned by running the server and reading what
it sent (see VERIFIED_VERSION), not from the schema alone.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.metadata
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ...types import events as events_
from ...types import messages as messages_
from ...types import usage as usage_
from .. import _bridge
from .. import _errors as errors
from . import _handoff, base, jsonrpc

KIND = "codex"
INSTALL = "npm install -g @openai/codex"
VERIFIED_VERSION = "0.160.1"
"""The CLI the live tests last passed against. Older ones get a warning."""
CRASHED = "__crashed__"
"""Put on the turn's queue when app-server dies."""
LIST_PAGE_SIZE = 100
LIST_PAGES = 50
"""`list_sessions()` reads at most this many pages of threads."""

APPROVAL_METHODS = (
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
    "item/permissions/requestApproval",
    "execCommandApproval",
    "applyPatchApproval",
)

TOOL_ITEMS = (
    "fileChange",
    "commandExecution",
    "mcpToolCall",
    "webSearch",
    "toolCall",
)
"""Item types that are the agent doing something, rather than speaking."""

AUTH_HINT = (
    "codex uses your own login: run `codex login`, or pass "
    "env={'OPENAI_API_KEY': ...}, or point it at another provider with "
    "config={'model_provider': ..., 'model_providers.<name>.base_url': ...}."
)


class CodexAdapter:
    def __init__(
        self,
        *,
        cwd: str,
        env: dict[str, str],
        executable: str,
        model: str | None,
        mcp_servers: dict[str, Any],
        approve: base.Approve | None,
        sandbox: str | None,
        config: dict[str, Any],
    ) -> None:
        # Codex records a thread's directory as its real path (on macOS a
        # temp dir is /var/..., recorded as /private/var/..., measured).
        self._cwd = os.path.realpath(cwd)
        self._env = env
        self._executable = executable
        self._model = model
        self._approve = approve
        # Full access by default, like claude without a gate: a worker is
        # driven by a program, and a mode that silently refuses writes reads
        # as an agent that did nothing.
        self._sandbox = sandbox or "danger-full-access"
        # Passed to app-server as `-c key=value`, the overrides the codex
        # CLI takes. An explicit config wins over the MCP servers.
        self._config = {**mcp_config(mcp_servers), **config}
        self._instructions: str | None = None
        self._process: base.Process | None = None
        self._peer: jsonrpc.JsonRpcPeer | None = None
        self._thread: str | None = None
        self._turn: str | None = None
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None
        self._usage = usage_.Usage()
        self._interrupted = False
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
        self._process = await base.spawn(
            [self._executable, "app-server", *_config_args(self._config)],
            cwd=self._cwd,
            env=self._env,
        )
        self._peer = jsonrpc.JsonRpcPeer(self._process, kind=KIND)
        self._peer.start(
            self._on_notify, self._on_request, on_crash=self._on_crash
        )
        try:
            await self._peer.call(
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
            await self._peer.notify("initialized")
        except BaseException:
            # A failed handshake must not leave the server running.
            await self.close()
            raise

    async def close(self) -> None:
        if self._peer is not None:
            await self._peer.close()
            self._peer = None
        if self._process is not None:
            with contextlib.suppress(Exception):
                await self._process.terminate()
            self._process = None

    def _rpc(self) -> jsonrpc.JsonRpcPeer:
        if self._peer is None:
            raise errors.WorkerError(f"{KIND}: app-server is not running")
        return self._peer

    def _thread_params(self) -> dict[str, Any]:
        """How the thread runs: on start, resume and fork alike.

        A resumed or forked thread otherwise keeps the policy it was created
        with, like a person's TUI asking for approvals nobody here answers.
        """
        return {
            "sandbox": self._sandbox,
            # With a gate: ask about everything the policy can ask about.
            # Without one the server never asks: a request nobody answers
            # would freeze the agent (measured).
            "approvalPolicy": "untrusted" if self._approve else "never",
            **({"model": self._model} if self._model else {}),
            **(
                {"developerInstructions": self._instructions}
                if self._instructions
                else {}
            ),
        }

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
        if session_id is not None:
            await self._require_here(session_id)
            method = "thread/fork" if fork else "thread/resume"
            try:
                result = await self._rpc().call(
                    method, {"threadId": session_id, **self._thread_params()}
                )
            except jsonrpc.JsonRpcError as exc:
                raise errors.WorkerError(
                    f"{KIND}: cannot open thread {session_id}: {exc}"
                ) from exc
            thread = str((result or {}).get("thread", {}).get("id") or "")
            self._thread = thread or session_id
            return self._thread
        result = await self._rpc().call(
            "thread/start", {"cwd": self._cwd, **self._thread_params()}
        )
        thread = str((result or {}).get("thread", {}).get("id") or "")
        if not thread:
            raise errors.WorkerError(
                f"{KIND}: thread/start gave no id: {result}"
            )
        self._thread = thread
        if seed:
            # The thread's model-visible history, before the first turn.
            _handoff.warn_unsupported_files(seed, KIND)
            await self._rpc().call(
                "thread/inject_items",
                {"threadId": thread, "items": _handoff.to_codex_items(seed)},
            )
        return thread

    async def _require_here(self, session_id: str) -> dict[str, Any]:
        """Refuse a thread that belongs to another directory.

        Codex's thread store is the whole machine's, so resuming a thread
        from another project succeeds natively, and the agent would then
        carry a transcript about files that are not here.
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
        if cwd is not None and cwd != self._cwd:
            raise errors.WorkerError(
                f"{KIND}: thread {session_id} belongs to {cwd}, not {self._cwd}"
            )
        return thread

    # -- notifications and approvals ------------------------------------------

    async def _on_notify(self, method: str, params: dict[str, Any]) -> None:
        thread = params.get("threadId") or (params.get("thread") or {}).get(
            "id"
        )
        if thread != self._thread:
            return
        if method == "thread/tokenUsage/updated":
            self._usage = _usage(params.get("tokenUsage") or {})
        if self._queue is not None:
            self._queue.put_nowait((method, params))

    def _on_crash(self, exc: BaseException) -> None:
        if self._queue is not None:
            self._queue.put_nowait((CRASHED, {"detail": str(exc)}))
        self._peer = None

    async def _on_request(
        self, request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        if method not in APPROVAL_METHODS or self._approve is None:
            # Unknown, or unasked for: answer rather than hang the agent.
            decision = "accept" if method in APPROVAL_METHODS else "decline"
            await self._rpc().respond(request_id, {"decision": decision})
            return
        granted, _ = await self._approve(
            str(params.get("itemId") or params.get("callId") or ""),
            _approval_tool_name(method),
            {
                k: v
                for k, v in params.items()
                if k not in ("threadId", "turnId", "itemId") and v is not None
            },
        )
        # The protocol has no field for a reason: codex only hears "declined".
        await self._rpc().respond(
            request_id, {"decision": "accept" if granted else "decline"}
        )

    # -- turns ----------------------------------------------------------------

    async def turn(self, prompt: base.Prompt) -> AsyncIterator[events_.Event]:
        base.check_files(prompt, KIND, ("image/",))
        # Unbounded: the reader fills it inline and must never wait on a
        # slow consumer.
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._queue = queue
        self._interrupted = False
        self._usage = usage_.Usage()
        open_text: set[str] = set()
        open_reasoning: set[str] = set()
        started: set[str] = set()
        # The last transport error codex reported while retrying: if the turn
        # then fails without a message of its own, the only explanation.
        last_error = ""
        try:
            try:
                begun = await self._rpc().call(
                    "turn/start",
                    {
                        "threadId": self._thread,
                        "input": await _input_items(prompt),
                    },
                )
            except jsonrpc.JsonRpcError as exc:
                raise errors.turn_failed(KIND, str(exc), AUTH_HINT) from exc
            # The turn id comes from the RESPONSE: this server sends no
            # `turn/started`, and steer and interrupt need the id.
            turn = ((begun or {}).get("turn") or {}).get("id")
            self._turn = turn
            yield events_.StreamStart()
            while True:
                method, params = await queue.get()
                if method == CRASHED:
                    raise errors.WorkerError(
                        f"{KIND} crashed: {params.get('detail')}"
                    )
                # An interrupted turn's notifications can still arrive when
                # the next one starts. Without this the new turn settled on
                # the OLD turn's completion and came back empty.
                other = params.get("turnId") or (params.get("turn") or {}).get(
                    "id"
                )
                if turn and other and other != turn:
                    continue
                if method == "item/agentMessage/delta":
                    item_id = params.get("itemId", "")
                    if item_id not in open_text:
                        open_text.add(item_id)
                        yield events_.TextStart(block_id=item_id)
                    yield events_.TextDelta(
                        chunk=params.get("delta", ""), block_id=item_id
                    )
                elif method in (
                    "item/reasoning/textDelta",
                    "item/reasoning/summaryTextDelta",
                ):
                    item_id = params.get("itemId", "")
                    if item_id not in open_reasoning:
                        open_reasoning.add(item_id)
                        yield events_.ReasoningStart(block_id=item_id)
                    yield events_.ReasoningDelta(
                        chunk=params.get("delta", ""), block_id=item_id
                    )
                elif method == "item/started":
                    item = params.get("item") or {}
                    if item.get("type") in TOOL_ITEMS:
                        started.add(item.get("id", ""))
                        yield events_.BuiltinToolStart(
                            tool_call_id=item.get("id", ""),
                            tool_name=_tool_name(item),
                        )
                elif method == "item/completed":
                    item = params.get("item") or {}
                    item_id = item.get("id", "")
                    kind = item.get("type")
                    if kind == "agentMessage" and item_id in open_text:
                        open_text.discard(item_id)
                        yield events_.TextEnd(block_id=item_id)
                    elif kind == "agentMessage" and item.get("text"):
                        # A message that was never streamed.
                        yield events_.TextStart(block_id=item_id)
                        yield events_.TextDelta(
                            chunk=item["text"], block_id=item_id
                        )
                        yield events_.TextEnd(block_id=item_id)
                    elif kind == "reasoning" and item_id in open_reasoning:
                        open_reasoning.discard(item_id)
                        yield events_.ReasoningEnd(block_id=item_id)
                    elif kind in TOOL_ITEMS:
                        for event in base.builtin_call(
                            item_id,
                            _tool_name(item),
                            json.dumps(_tool_args(item)),
                            started=started,
                        ):
                            yield event
                        yield base.builtin_result(
                            item_id,
                            _tool_name(item),
                            _item_result(item),
                            is_error=item.get("status") == "failed",
                        )
                elif method == "turn/completed":
                    completed = params.get("turn") or {}
                    # A failed turn says so. Reported as a normal stop with
                    # empty text, "401 Unauthorized" became silence.
                    if (
                        completed.get("status") == "failed"
                        and not self._interrupted
                    ):
                        raise errors.turn_failed(
                            KIND,
                            _error_message(completed) or last_error,
                            AUTH_HINT,
                        )
                    yield events_.StreamEnd(
                        usage=self._usage,
                        finish_reason=(
                            "cancelled" if self._interrupted else "stop"
                        ),
                    )
                    return
                elif method == "turn/failed":
                    raise errors.turn_failed(
                        KIND, _error_message(params), AUTH_HINT
                    )
                elif method == "error":
                    last_error = _error_message(params) or last_error
        finally:
            self._queue = None
            self._turn = None

    async def steer(self, text: str) -> None:
        if not self._turn:
            raise errors.WorkerError(f"{KIND}: no turn is running to steer")
        try:
            await self._rpc().call(
                "turn/steer",
                {
                    "threadId": self._thread,
                    # A precondition: steering the wrong turn is worse than
                    # failing to steer.
                    "expectedTurnId": self._turn,
                    "input": [{"type": "text", "text": text}],
                },
            )
        except jsonrpc.JsonRpcError as exc:
            raise errors.WorkerError(f"{KIND}: steer rejected: {exc}") from exc

    async def interrupt(self) -> None:
        self._interrupted = True
        if not self._turn or self._peer is None:
            return
        with contextlib.suppress(Exception):
            await self._peer.call(
                "turn/interrupt",
                {"threadId": self._thread, "turnId": self._turn},
                timeout=30,
            )

    # -- the store ------------------------------------------------------------

    async def history(self, session_id: str) -> list[messages_.Message]:
        """Read the thread's stored record, every item included."""
        await self._require_here(session_id)
        result = await self._rpc().call(
            "thread/read",
            {"threadId": session_id, "includeTurns": True, "itemsView": "full"},
        )
        stored: list[messages_.Message] = []
        for turn in ((result or {}).get("thread") or {}).get("turns") or []:
            for item in turn.get("items") or []:
                stored.extend(_item_to_messages(item))
        return _handoff.model_view(stored)

    async def list_sessions(self) -> list[base.SessionInfo]:
        # Asked for THIS directory, page by page: one unfiltered page of the
        # machine's store (measured: 3159 threads) missed threads from here.
        threads: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(LIST_PAGES):
            page = await self._rpc().call(
                "thread/list",
                {
                    "cwd": self._cwd,
                    "limit": LIST_PAGE_SIZE,
                    **({"cursor": cursor} if cursor else {}),
                },
            )
            threads += (page or {}).get("data") or []
            cursor = (page or {}).get("nextCursor")
            if not cursor:
                break
        return [
            base.SessionInfo(
                session_id=t.get("id", ""),
                title=t.get("name") or t.get("preview") or None,
                cwd=t.get("cwd"),
                created_at=t.get("createdAt"),
                updated_at=t.get("updatedAt"),
            )
            for t in threads
            if t.get("id") and t.get("cwd") in (None, self._cwd)
        ]

    def tui_command(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        config = {
            k: v
            for k, v in self._config.items()
            if not k.startswith(f"mcp_servers.{_bridge.SERVER_NAME}.")
        }
        # --disable in_app_updates: measured, the TUI otherwise opens on an
        # "Update now / Skip" dialog that the first keystrokes answer.
        argv = [
            self._executable,
            *(["resume", session_id] if session_id else []),
            "--disable",
            "in_app_updates",
            *_config_args(config),
        ]
        if self._model:
            argv += ["--model", self._model]
        if self._instructions:
            argv += _config_args({"developer_instructions": self._instructions})
        # None: codex mints its own thread id.
        return argv, dict(self._env), None


async def _input_items(prompt: base.Prompt) -> list[dict[str, Any]]:
    """Return a prompt as app-server's `UserInput` items.

    An image travels as a data URL, built here from its bytes.
    """
    items: list[dict[str, Any]] = []
    for part in prompt:
        if isinstance(part, messages_.TextPart):
            items.append({"type": "text", "text": part.text})
        elif part.media_type.startswith("text/"):
            items.append({"type": "text", "text": await base.as_text(part)})
        else:
            encoded = base64.b64encode(await base.data(part)).decode("ascii")
            items.append(
                {
                    "type": "image",
                    "url": f"data:{part.media_type};base64,{encoded}",
                }
            )
    return items


def _tool_name(item: dict[str, Any]) -> str:
    # Never the command line: that made every shell call a differently named
    # tool. The command is in the arguments (`_tool_args`).
    return (
        item.get("toolName") or item.get("server") or item.get("type") or "tool"
    )


def _approval_tool_name(method: str) -> str:
    """Return the native tool kind an approval is about, never its args."""
    if "commandExecution" in method or "execCommand" in method:
        return "commandExecution"
    if "fileChange" in method or "applyPatch" in method:
        return "fileChange"
    return method.rsplit("/", 1)[-1]


def _tool_args(item: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in item.items()
        if k in ("command", "changes", "arguments", "query", "path", "cwd")
    }


def _item_result(item: dict[str, Any]) -> Any:
    """Return what a tool item produced, under whichever name codex uses.

    A command's output is `aggregatedOutput` (codex 0.160; reading only
    `output` made every command result None), a file change's `changes`, an
    MCP call's `result`.
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


def _item_to_messages(item: dict[str, Any]) -> list[messages_.Message]:
    """One stored thread item -> messages, CLI tools as tool calls."""
    kind = item.get("type")
    if kind == "userMessage":
        # A user item carries `content` blocks, its `UserInput`s.
        text = item.get("text")
        files: list[messages_.Part] = []
        if not isinstance(text, str):
            blocks = [
                b for b in item.get("content") or [] if isinstance(b, dict)
            ]
            text = "".join(b.get("text", "") for b in blocks)
            for block in blocks:
                url = block.get("url")
                if block.get("type") == "image" and isinstance(url, str):
                    files.append(messages_.FilePart.from_url(url))
        return [
            messages_.Message(
                role="user", parts=[messages_.TextPart(text=text), *files]
            )
        ]
    if kind == "agentMessage" and item.get("text"):
        return [
            messages_.Message(
                role="assistant", parts=[messages_.TextPart(text=item["text"])]
            )
        ]
    if kind == "reasoning" and _reasoning_text(item):
        return [
            messages_.Message(
                role="assistant",
                parts=[messages_.ReasoningPart(text=_reasoning_text(item))],
            )
        ]
    if kind in TOOL_ITEMS:
        call = messages_.ToolCallPart(
            tool_call_id=item.get("id", ""),
            tool_name=_tool_name(item),
            tool_args=json.dumps(_tool_args(item)),
        )
        result = messages_.ToolResultPart(
            tool_call_id=item.get("id", ""),
            tool_name=call.tool_name,
            result=_item_result(item),
            result_kind="error" if item.get("status") == "failed" else "json",
        )
        return [
            messages_.Message(role="assistant", parts=[call]),
            messages_.Message(role="tool", parts=[result]),
        ]
    return []


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


def mcp_config(servers: dict[str, Any]) -> dict[str, Any]:
    """MCP servers, in the shape claude takes, as codex `-c` overrides.

    `{"name": {"command": ..., "args": [...], "env": {...}}}` for a local
    server, `{"name": {"url": ...}}` for a remote one.
    """
    out: dict[str, Any] = {}
    for name, server in servers.items():
        prefix = f"mcp_servers.{name}"
        for key in ("command", "args", "url"):
            if server.get(key) is not None:
                out[f"{prefix}.{key}"] = server[key]
        for var, value in (server.get("env") or {}).items():
            out[f"{prefix}.env.{var}"] = value
    return out


def _config_args(config: dict[str, Any]) -> list[str]:
    """`-c key=value` pairs, the values as TOML.

    JSON string, number and bool literals are valid TOML.
    """
    args: list[str] = []
    for key, value in config.items():
        args += ["-c", f"{key}={json.dumps(value)}"]
    return args
