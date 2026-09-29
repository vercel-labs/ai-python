"""A small JSON-RPC 2.0 peer over a workspace process's stdio.

Deliberately minimal: line-delimited JSON, request/response futures, and
two callbacks for the things a server can initiate. Nothing here is
protocol-specific — the Codex adapter supplies the meaning.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .. import errors

if TYPE_CHECKING:
    from ....workspaces.experimental import _base

NotifyHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
RequestHandler = Callable[[Any, str, dict[str, Any]], Awaitable[None]]


class JsonRpcPeer:
    def __init__(self, process: _base.Process, *, harness: str) -> None:
        self._process = process
        self._harness = harness
        self._next_id = 0
        self._waiters: dict[int, asyncio.Future[Any]] = {}
        self._pump: asyncio.Task[None] | None = None
        self._closed = False
        self._on_crash: Callable[[BaseException], None] | None = None

    def start(
        self,
        on_notify: NotifyHandler,
        on_request: RequestHandler,
        on_crash: Callable[[BaseException], None] | None = None,
    ) -> None:
        # `on_crash` is for listeners, not callers. A turn does not await a
        # request/response future; it waits on a stream of notifications.
        # When the process dies, failing the futures wakes nobody who was
        # listening — measured: a SIGKILLed codex left the turn blocked for
        # the full 300s test timeout, with no AgentCrashedError ever raised.
        self._on_crash = on_crash
        self._pump = asyncio.create_task(self._read_loop(on_notify, on_request))

    async def close(self) -> None:
        self._closed = True
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
            self._pump = None
        # Nobody is coming with an answer; wake every caller rather than
        # leaving them awaiting a future that can never resolve. A close is
        # not a crash: the owner is shutting down on purpose and still holds
        # the process to terminate, so `on_crash` must NOT fire here — when
        # it did, the adapter dropped its process handle and close() never
        # terminated the server, leaving it alive holding thread locks.
        self._fail_pending(
            errors.AgentCrashedError(self._harness, "connection closed"),
            crashed=False,
        )

    def _fail_pending(
        self, exc: BaseException, *, crashed: bool = True
    ) -> None:
        for future in list(self._waiters.values()):
            if not future.done():
                future.set_exception(exc)
        self._waiters.clear()
        if crashed and self._on_crash is not None:
            with contextlib.suppress(Exception):
                self._on_crash(exc)

    async def call(
        self, method: str, params: Any = None, *, timeout: float = 120
    ) -> Any:
        if self._closed:
            raise errors.AgentCrashedError(
                self._harness, "connection is closed"
            )
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = future
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                **({"params": params} if params is not None else {}),
            }
        )
        try:
            async with asyncio.timeout(timeout):
                return await future
        finally:
            self._waiters.pop(request_id, None)

    async def notify(self, method: str, params: Any = None) -> None:
        await self._send(
            {
                "jsonrpc": "2.0",
                "method": method,
                **({"params": params} if params is not None else {}),
            }
        )

    async def respond(self, request_id: Any, result: Any) -> None:
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _send(self, message: dict[str, Any]) -> None:
        try:
            await self._process.write(json.dumps(message) + "\n")
        except Exception as exc:
            raise errors.AgentCrashedError(
                self._harness, f"could not write to the agent: {exc}"
            ) from exc

    async def _read_loop(
        self, on_notify: NotifyHandler, on_request: RequestHandler
    ) -> None:
        try:
            while True:
                line = await self._process.readline()
                if not line:
                    # EOF: the agent is gone. Everyone waiting must hear it.
                    self._fail_pending(
                        errors.AgentCrashedError(
                            self._harness, await self._death_note()
                        )
                    )
                    return
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a stray non-JSON line is noise, not a failure
                if "method" in message and "id" not in message:
                    await on_notify(
                        message["method"], message.get("params") or {}
                    )
                elif "method" in message:
                    await on_request(
                        message["id"],
                        message["method"],
                        message.get("params") or {},
                    )
                else:
                    self._resolve(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(
                errors.AgentCrashedError(self._harness, str(exc))
            )

    async def _death_note(self) -> str:
        """Describe the agent's last words AND how it died.

        So a crash is diagnosable.

        The exit code matters as much as stderr: a process killed by the
        kernel (OOM → 137 / -9) writes nothing, and a note made of stderr
        alone then shows only some unrelated startup warning as the "cause".
        """
        code: int | None = None
        tail = ""
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2):
                code = await self._process.wait()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2):
                tail = (await self._process.read_stderr()).strip()
        how = f"exit code {code}" if code is not None else "exited"
        if code in (137, -9):
            how += " (SIGKILL — typically the kernel's OOM killer)"
        return f"{how}; stderr: {tail[-800:]}" if tail else f"{how}; no stderr"

    def _resolve(self, message: dict[str, Any]) -> None:
        msg_id = message.get("id")
        future = (
            self._waiters.pop(msg_id, None) if isinstance(msg_id, int) else None
        )
        if future is None or future.done():
            return
        if "error" in message:
            future.set_exception(JsonRpcError(message["error"]))
        else:
            future.set_result(message.get("result"))


class JsonRpcError(Exception):
    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error.get("message") or json.dumps(error))
        self.code = error.get("code")
        self.data = error.get("data")
