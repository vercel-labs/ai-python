"""Fx ACP v1 over JSON-RPC stdio. Shapes verified with fx 0.0.17."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
import pathlib
import shutil
import warnings
from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING, Any, Literal

from ...types import events as events_
from ...types import messages as messages_
from ...types import usage as usage_
from .. import _bridge, _jsonrpc, _process, _utils, _wire
from .. import _errors as errors
from . import handoff

if TYPE_CHECKING:
    from .harness import Fx

KIND = "fx"
INSTALL = "curl -fsSL https://fx.sh/setup.sh | bash"
VERIFIED_VERSION = "0.0.17"
AUTH_HINT = "fx uses your own login: run `fx login`, or set AI_GATEWAY_API_KEY."
AUTH_MARKERS = (
    "fx needs access",
    "fx needs a",
    "credential refresh failed",
    "authentication failed",
    "run fx login",
)


class FxWire:
    def __init__(self, harness: Fx) -> None:
        self.harness = harness
        self.cwd = os.path.realpath(harness.cwd)
        self.process: _process.Process | None = None
        self.peer: _jsonrpc.JsonRpcPeer | None = None
        self.session_id = ""
        self.version = VERIFIED_VERSION
        self.model: str | None = None
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self.running: int | None = None
        self.idle = asyncio.Event()
        self.idle.set()
        self.interrupted = False
        # A request stays busy until its response, not until its content ends.
        self.requests: dict[int, _Prompt] = {}
        self.order = 0
        self.retries: list[_Prompt] = []
        self.retry_task: asyncio.Task[None] | None = None
        self.loading: set[str] | None = None
        self.writing = False
        self.early: list[tuple[str, Any]] = []
        self.closed = False
        self.failure: errors.ExternalHarnessError | None = None

    def _rpc(self) -> _jsonrpc.JsonRpcPeer:
        if self.closed:
            raise errors.ExternalHarnessError("fx: connection is closed")
        if self.failure is not None:
            raise self.failure
        if self.peer is None:
            raise errors.ExternalHarnessError("fx: ACP is not running")
        return self.peer

    async def start(
        self,
        *,
        session_id: str | None,
        bridge_url: str,
        bridge_token: str,
        instructions: str | None,
        seed: list[messages_.Message],
    ) -> str:
        harness = self.harness
        executable = harness.provider.executable or "fx"
        await _process.probe(
            KIND,
            executable,
            install=INSTALL,
            verified=VERIFIED_VERSION,
            cwd=self.cwd,
            env=harness.provider.env,
        )
        argv = [executable, "acp", "--permission-mode", "full-access"]
        for flag, value in (
            ("--provider", harness.model_provider),
            ("--model", harness.model_name),
            ("--effort", harness.effort),
        ):
            if value is not None:
                argv.extend((flag, value))
        self.process = await _process.spawn(
            argv, cwd=self.cwd, env=harness.provider.env
        )
        self.peer = _jsonrpc.JsonRpcPeer(self.process, kind=KIND)
        self.peer.start(self._on_notify, self._on_request, self._on_crash)
        try:
            try:
                initialized = await self.peer.call(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {"session": {"notices": {}}},
                    },
                )
            except _jsonrpc.JsonRpcError as exc:
                raise errors.turn_failed(
                    KIND, str(exc), AUTH_HINT, AUTH_MARKERS
                ) from exc
            self.version = (
                initialized.get("agentInfo", {}).get("version") or self.version
            )
            params: dict[str, Any] = {
                "cwd": self.cwd,
                "mcpServers": [
                    {
                        "type": "http",
                        "name": _bridge.SERVER_NAME,
                        "url": bridge_url,
                        "headers": [
                            {
                                "name": "Authorization",
                                "value": f"Bearer {bridge_token}",
                            }
                        ],
                    }
                ],
            }
            if session_id is not None:
                cursor = None
                while True:
                    page = await self.peer.call(
                        "session/list",
                        {
                            "cwd": self.cwd,
                            **({"cursor": cursor} if cursor else {}),
                        },
                    )
                    if any(
                        s.get("sessionId") == session_id
                        and os.path.realpath(s.get("cwd", self.cwd)) == self.cwd
                        for s in page.get("sessions", [])
                    ):
                        break
                    cursor = page.get("nextCursor")
                    if not cursor:
                        raise errors.ExternalHarnessError(
                            f"fx: session {session_id} is unknown or belongs "
                            f"to another directory, not {self.cwd}"
                        )
                self.session_id = session_id
                result = await self.peer.call(
                    "session/resume", {"sessionId": session_id, **params}
                )
            else:
                result = await self.peer.call(
                    "session/new",
                    {
                        **params,
                        **(
                            {
                                "systemPrompt": [
                                    {"type": "text", "text": instructions}
                                ]
                            }
                            if instructions
                            else {}
                        ),
                    },
                )
                if seed:
                    values = {
                        o["id"]: o.get("currentValue")
                        for o in result.get("configOptions", [])
                    }
                    prefs = {
                        k: values.get(k)
                        for k in ("provider", "model", "effort")
                    }
                    prefs["fast_mode"] = values.get("fast") in (
                        True,
                        "true",
                        "on",
                    )
                    home = (
                        harness.provider.env.get("FX_HOME")
                        or os.environ.get("FX_HOME")
                        or str(pathlib.Path.home() / ".fx")
                    )
                    _utils.warn_unsupported_files(
                        seed,
                        KIND,
                        accepted_media_types=("image/",),
                        accepted_description="images",
                    )
                    dropped = _utils.count_reasoning_parts(seed)
                    if dropped:
                        warnings.warn(
                            f"{dropped} reasoning part(s) in the history "
                            "were not carried into fx: no portable reasoning "
                            "format. Text and tool calls were carried.",
                            stacklevel=2,
                        )
                    self.session_id, count = await asyncio.to_thread(
                        handoff.write_session,
                        seed,
                        fx_home=home,
                        cwd=self.cwd,
                        prefs=prefs,
                        instructions=instructions,
                    )
                    self.loading = set()
                    try:
                        result = await self.peer.call(
                            "session/load",
                            {"sessionId": self.session_id, **params},
                        )
                        if len(self.loading) != count:
                            raise ValueError(
                                f"replayed {len(self.loading)} user messages, "
                                f"expected {count}"
                            )
                    except BaseException as exc:
                        with contextlib.suppress(Exception):
                            await self.peer.call(
                                "session/close",
                                {"sessionId": self.session_id},
                                timeout=2,
                            )
                        shutil.rmtree(
                            pathlib.Path(home)
                            / "sessions"
                            / "v2"
                            / self.session_id,
                            ignore_errors=True,
                        )
                        self.session_id = ""
                        if not isinstance(exc, Exception):
                            raise
                        raise errors.ExternalHarnessError(
                            f"fx {self.version}: this version of ai cannot "
                            f"seed an fx session: {exc}"
                        ) from exc
                    finally:
                        self.loading = None
                else:
                    self.session_id = str(result.get("sessionId") or "")
            if not self.session_id:
                raise errors.ExternalHarnessError(
                    "fx: session/new gave no session id"
                )
            self.model = next(
                (
                    o.get("currentValue")
                    for o in result.get("configOptions", [])
                    if o.get("id") == "model"
                ),
                None,
            )
            return self.session_id
        except BaseException:
            await self.close()
            raise

    async def send(self, message: messages_.Message) -> None:
        self._rpc()
        _utils.raise_for_unsupported_parts(message, KIND, ("image/",))
        prompt: list[dict[str, Any]] = []
        text_only = True
        for part in message.parts:
            if isinstance(part, messages_.TextPart):
                prompt.append({"type": "text", "text": part.text})
            elif isinstance(part, messages_.FilePart):
                text_only = False
                if part.media_type.startswith("text/"):
                    prompt.append(
                        {
                            "type": "resource",
                            "resource": {
                                "uri": "file:///"
                                + (part.filename or "attached-file"),
                                "text": (
                                    await _utils.extract_data(part)
                                ).decode(errors="replace"),
                            },
                        }
                    )
                else:
                    prompt.append(
                        {
                            "type": "image",
                            "mimeType": part.media_type,
                            "data": base64.b64encode(
                                await _utils.extract_data(part)
                            ).decode("ascii"),
                        }
                    )
        while True:
            self._rpc()
            if self.retry_task is not None:
                try:
                    await asyncio.shield(self.retry_task)
                except asyncio.CancelledError:
                    self._rpc()  # A close must fail send(), not cancel it.
                    raise
            elif not text_only and self.requests:
                await self.idle.wait()
            else:
                break
        self.order += 1
        await self._prompt(_Prompt(self.order, message, prompt))

    async def _prompt(self, record: _Prompt) -> None:
        steer = bool(self.requests)
        params = {
            "sessionId": self.session_id,
            "prompt": record.prompt,
            **({"_meta": {"fx": {"steer": True}}} if steer else {}),
        }

        def prepared(request_id: int) -> None:
            self.requests[request_id] = record
            self.idle.clear()
            if not steer:
                self.running = request_id
                self.interrupted = False
                record.accepted = True

        if not steer:
            self.writing = True
        try:
            await self._rpc()._begin_call(
                "session/prompt",
                params,
                prepared=prepared,
                completed=self._finished,
            )
            if not steer:
                self.queue.put_nowait(("accepted", record.message))
        except BaseException:
            # A failed write must not leave a request that can never resolve.
            for key, value in list(self.requests.items()):
                if value is record:
                    self.requests.pop(key)
            if not self.requests:
                self.idle.set()
            raise
        finally:
            if not steer:
                self.writing = False
                for item in self.early:
                    self.queue.put_nowait(item)
                self.early.clear()

    def _emit(self, kind: str, value: Any) -> None:
        if self.writing:
            self.early.append((kind, value))
        else:
            self.queue.put_nowait((kind, value))

    def _accept(self, request_id: int, record: _Prompt) -> None:
        if not record.accepted:
            record.accepted = True
            self._emit("accepted", record.message)
        if self.running is None:
            self.running = request_id

    def _finished(self, request_id: int, future: asyncio.Future[Any]) -> None:
        record = self.requests.pop(request_id, None)
        if record is None:
            if not future.cancelled():
                future.exception()
            return
        if not self.requests:
            self.idle.set()
        if future.cancelled():
            return
        exc = future.exception()
        result = {} if exc else (future.result() or {})
        outcome = (result.get("_meta") or {}).get("fx", {}).get("steering")
        # Once a request owns a turn, its response always ends that turn.
        # Accepted messages are never retried, whatever the steering outcome.
        if self.running != request_id:
            if not record.accepted and (outcome == "dropped" or exc):
                self.retries.append(record)
                if self.retry_task is None and not self.closed:
                    self.retry_task = asyncio.create_task(self._retry())
                return
            if outcome == "absorbed" or record.accepted:
                if not record.accepted:
                    self._accept(request_id, record)
            else:
                self._accept(request_id, record)
            if self.running != request_id:
                return
        self.running = None
        self.interrupted = (
            self.interrupted or result.get("stopReason") == "cancelled"
        )
        self._emit("end", (result, exc, self.interrupted))

    async def _retry(self) -> None:
        try:
            while self.retries and not self.closed:
                await self.idle.wait()
                self._rpc()
                self.retries.sort(key=lambda record: record.order)
                record = self.retries.pop(0)
                await self._prompt(record)
        except Exception as exc:
            self._on_crash(exc)
        finally:
            self.retry_task = None

    async def _on_notify(self, method: str, params: dict[str, Any]) -> None:
        if method != "session/update":
            return
        update = params.get("update") or {}
        if self.loading is not None:
            if update.get("sessionUpdate") == "user_message_chunk":
                self.loading.add(str(update.get("messageId")))
            return
        if params.get("sessionId") != self.session_id:
            return
        # After a cancel, late updates cannot start a waiting steer's turn.
        # Its response decides whether it was dropped or ran on its own.
        if self.running is None and self.interrupted:
            return
        kind = update.get("sessionUpdate")
        if kind == "user_message_chunk":
            request_id = (
                (update.get("_meta") or {})
                .get("fx", {})
                .get("steering", {})
                .get("requestId")
            )
            record = self.requests.get(request_id)
            if record is not None:
                self._accept(request_id, record)
        elif self.running is None and kind in (
            "agent_message_chunk",
            "agent_thought_chunk",
            "tool_call",
        ):
            candidate = next(
                (
                    (key, value)
                    for key, value in self.requests.items()
                    if not value.accepted
                ),
                None,
            )
            if candidate is not None:
                self._accept(*candidate)
        if self.running is not None:
            self._emit("update", update)

    async def _on_request(
        self, request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        if method == "session/request_permission":
            options = params.get("options") or []
            option = next(
                (
                    o["optionId"]
                    for o in options
                    if o.get("kind") in ("allow_once", "allow_always")
                    or o.get("optionId") == "allow_once"
                ),
                None,
            )
            outcome = (
                {"outcome": "selected", "optionId": option}
                if option
                else {"outcome": "cancelled"}
            )
            await self._rpc().respond(request_id, {"outcome": outcome})
        else:
            raise ValueError(f"fx: unsupported server request {method}")

    def _on_crash(self, exc: BaseException) -> None:
        if self.closed or self.failure is not None:
            return
        self.failure = errors.ExternalHarnessError(str(exc))
        self.requests.clear()
        self.retries.clear()
        self.running = None
        self.idle.set()
        self.queue.put_nowait(("crashed", self.failure))

    async def interrupt(self) -> None:
        if self.requests and self.peer is not None:
            self.interrupted = True
            await self.peer.notify(
                "session/cancel", {"sessionId": self.session_id}
            )

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.requests.clear()
        self.retries.clear()
        self.running = None
        self.idle.set()
        task = self.retry_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.peer is not None:
            if self.session_id:
                with contextlib.suppress(Exception):
                    await self.peer.call(
                        "session/close",
                        {"sessionId": self.session_id},
                        timeout=2,
                    )
            await self.peer.close()
            self.peer = None
        if self.process is not None:
            with contextlib.suppress(Exception):
                await self.process.terminate()
            self.process = None

    async def events(self) -> AsyncGenerator[_wire.Item]:
        state = _TurnState()
        while True:
            kind, value = await self.queue.get()
            if kind == "crashed":
                raise errors.ExternalHarnessError(str(value))
            if kind == "accepted":
                yield _wire.Accepted(value)
            elif kind == "update":
                for item in state.translate(value):
                    yield item
            elif kind == "end":
                for item in state.close_block():
                    yield item
                result, exc, cancelled = value
                stop = result.get("stopReason")
                reason: Literal["stop", "length", "interrupted", "error"] = (
                    "stop"
                )
                error = None
                if cancelled or stop == "cancelled":
                    reason = "interrupted"
                elif exc is not None or stop == "refused":
                    reason = "error"
                    error = errors.turn_failed(
                        KIND,
                        str(exc) if exc else state.recovery or state.last_text,
                        AUTH_HINT,
                        AUTH_MARKERS,
                    )
                elif stop in ("max_output_tokens", "max_model_turns"):
                    reason = "length"
                raw = result.get("usage")
                usage = (
                    None
                    if raw is None
                    else usage_.Usage(
                        input_tokens=raw.get("inputTokens") or 0,
                        output_tokens=raw.get("outputTokens") or 0,
                        cache_read_tokens=raw.get("cacheReadTokens"),
                        cache_write_tokens=raw.get("cacheWriteTokens"),
                        reasoning_tokens=raw.get("reasoningTokens"),
                        raw=raw,
                    )
                )
                yield _wire.TurnEnd(
                    reason=reason,
                    error=error,
                    usage=usage,
                    metadata={"model": self.model, "fx_version": self.version},
                )
                state = _TurnState()


@dataclasses.dataclass
class _Prompt:
    order: int
    message: messages_.Message
    prompt: list[dict[str, Any]]
    accepted: bool = False


class _TurnState:
    def __init__(self) -> None:
        self.block: tuple[str, str] | None = None
        self.counter = 0
        self.tools: dict[str, str] = {}
        self.started: set[str] = set()
        self.completed: set[str] = set()
        self.last_text = ""
        self.recovery = ""

    def close_block(self) -> Iterator[_wire.Content]:
        if self.block is not None:
            kind, block_id = self.block
            self.block = None
            if kind == "agent_message_chunk":
                yield events_.TextEnd(block_id=block_id)
            else:
                yield events_.ReasoningEnd(block_id=block_id)

    def translate(self, update: dict[str, Any]) -> Iterator[_wire.Content]:
        kind = update.get("sessionUpdate")
        if kind in ("agent_message_chunk", "agent_thought_chunk"):
            block_id = (
                str(update.get("messageId") or "")
                if kind == "agent_message_chunk"
                else (
                    self.block[1]
                    if self.block and self.block[0] == kind
                    else f"reasoning-{self.counter}"
                )
            )
            if self.block != (kind, block_id):
                yield from self.close_block()
                self.counter += 1
                self.block = (kind, block_id)
                if kind == "agent_message_chunk":
                    self.last_text = ""
                    yield events_.TextStart(block_id=block_id)
                else:
                    yield events_.ReasoningStart(block_id=block_id)
            text = (update.get("content") or {}).get("text", "")
            if kind == "agent_message_chunk":
                self.last_text += text
                yield events_.TextDelta(block_id=block_id, chunk=text)
            else:
                yield events_.ReasoningDelta(block_id=block_id, chunk=text)
            return
        if kind in ("tool_call", "tool_call_update", "user_message_chunk"):
            yield from self.close_block()
        call_id = str(update.get("toolCallId") or "")
        if kind == "tool_call":
            native = (
                (update.get("_meta") or {}).get("fx", {}).get("toolCall", {})
            )
            if native.get("mcp", {}).get("server") == _bridge.SERVER_NAME:
                return
            name = str(update.get("name") or "tool")
            self.tools[call_id] = name
            args = update.get("rawInput") or {}
            yield from _utils.get_builtin_call_events(
                call_id,
                name,
                args if isinstance(args, str) else json.dumps(args),
                started_call_ids=self.started,
            )
        elif kind == "tool_call_update" and update.get("status") in (
            "completed",
            "failed",
        ):
            if call_id not in self.tools or call_id in self.completed:
                return
            self.completed.add(call_id)
            text = "".join(
                (p.get("content") or {}).get("text", "")
                for p in update.get("content", [])
            )
            yield _utils.get_builtin_result_event(
                call_id,
                self.tools.get(call_id, "tool"),
                text,
                is_error=update.get("status") == "failed",
            )
        elif kind == "session_info_update":
            recovery = (update.get("_meta") or {}).get("fx", {}).get(
                "modelResponseRecovery"
            ) or {}
            self.recovery = recovery.get("message") or self.recovery
