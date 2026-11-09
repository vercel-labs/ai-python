"""The transport to `codex app-server`: its process, and JSON-RPC over stdio.

Line-delimited JSON, request/response futures, and two callbacks for what
a server can start. Nothing here is protocol-specific: the Codex adapter
supplies the meaning.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from .. import _errors as errors

# Longest line app-server may write. One JSON line can be megabytes (a
# `thread/read` of a long thread). With asyncio's default (64 KiB) such a line
# failed as if the CLI had crashed.
LINE_LIMIT = 256 * 1024 * 1024

# Seconds app-server gets to exit after SIGTERM, before SIGKILL.
TERM_GRACE = 3.0

NotifyHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
RequestHandler = Callable[[Any, str, dict[str, Any]], Awaitable[None]]


class JsonRpcError(Exception):
    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error.get("message") or json.dumps(error))
        self.code = error.get("code")
        self.data = error.get("data")


class JsonRpcPeer:
    def __init__(self, process: Process, *, kind: str) -> None:
        self._process = process
        self._kind = kind
        self._next_id = 0
        self._waiters: dict[int, asyncio.Future[Any]] = {}
        self._pump: asyncio.Task[None] | None = None
        self._closed = False
        self._on_crash: Callable[[BaseException], None] | None = None
        # Server requests being answered. Each runs as its own task: an
        # answer can take minutes (a person approving a tool call), and the
        # reader must keep reading meanwhile.
        self._requests: set[asyncio.Task[None]] = set()
        self._answered: set[Any] = set()

    def start(
        self,
        on_notify: NotifyHandler,
        on_request: RequestHandler,
        on_crash: Callable[[BaseException], None] | None = None,
    ) -> None:
        # `on_crash` is for listeners. A turn waits on a stream of
        # notifications, not on a future: when the process dies, failing
        # the futures wakes nobody who was listening (measured: a SIGKILLed
        # codex left the turn blocked until the test timed out).
        self._on_crash = on_crash
        self._pump = asyncio.create_task(self._read_loop(on_notify, on_request))

    async def close(self) -> None:
        self._closed = True
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
            self._pump = None
        for task in list(self._requests):
            task.cancel()
        for task in list(self._requests):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._requests.clear()
        # A close is not a crash: `on_crash` does not fire.
        self._fail_pending(
            errors.WorkerError(f"{self._kind}: connection closed"),
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
            raise errors.WorkerError(f"{self._kind}: connection is closed")
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
        self._answered.add(request_id)
        await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _answer(
        self,
        on_request: RequestHandler,
        request_id: Any,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Run one request handler; never leave the server without a reply."""
        try:
            await on_request(request_id, method, params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if request_id not in self._answered:
                with contextlib.suppress(Exception):
                    await self._send(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32603, "message": str(exc)},
                        }
                    )
        finally:
            self._answered.discard(request_id)

    async def _send(self, message: dict[str, Any]) -> None:
        try:
            await self._process.write(json.dumps(message) + "\n")
        except Exception as exc:
            raise errors.WorkerError(
                f"{self._kind}: could not write to the CLI: {exc}"
            ) from exc

    async def _read_loop(
        self, on_notify: NotifyHandler, on_request: RequestHandler
    ) -> None:
        try:
            while True:
                line = await self._process.readline()
                if not line:
                    note = await self._process.death_note()
                    self._fail_pending(
                        errors.WorkerError(f"{self._kind} crashed: {note}")
                    )
                    return
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a stray non-JSON line is noise
                if not isinstance(message, dict):
                    continue
                if "method" in message and "id" not in message:
                    # Inline, so notifications are handled in the order sent.
                    await on_notify(
                        message["method"], message.get("params") or {}
                    )
                elif "method" in message:
                    task = asyncio.create_task(
                        self._answer(
                            on_request,
                            message["id"],
                            message["method"],
                            message.get("params") or {},
                        )
                    )
                    self._requests.add(task)
                    task.add_done_callback(self._requests.discard)
                else:
                    self._resolve(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(errors.WorkerError(f"{self._kind}: {exc}"))

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


# -- the process -------------------------------------------------------------


class Process:
    """A CLI started by `spawn`: JSON lines on stdin and stdout."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    async def write(self, data: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()

    async def readline(self) -> str:
        assert self._proc.stdout is not None
        return (await self._proc.stdout.readline()).decode()

    async def read_stderr(self) -> str:
        if self._proc.stderr is None:
            return ""
        return (await self._proc.stderr.read()).decode(errors="replace")

    async def wait(self) -> int:
        return await self._proc.wait()

    async def terminate(self) -> None:
        """Stop the process group: SIGTERM, then SIGKILL."""
        if self._proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self._proc.pid), 15)
        try:
            async with asyncio.timeout(TERM_GRACE):
                await self._proc.wait()
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(self._proc.pid), 9)
            with contextlib.suppress(Exception):
                await self._proc.wait()

    async def death_note(self) -> str:
        """Say how the process ended, and its last words."""
        code: int | None = None
        tail = ""
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2):
                code = await self.wait()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(2):
                tail = (await self.read_stderr()).strip()
        how = f"exit code {code}" if code is not None else "exited"
        if code in (137, -9):
            how += " (SIGKILL, typically the kernel's OOM killer)"
        return f"{how}; stderr: {tail[-800:]}" if tail else f"{how}; no stderr"


async def spawn(
    argv: Sequence[str], *, cwd: str, env: Mapping[str, str]
) -> Process:
    """Start a CLI in its own process group, with pipes for stdio."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env={**os.environ, **env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Own the whole tree, so teardown reaches grandchildren too.
            start_new_session=True,
            limit=LINE_LIMIT,
        )
    except OSError as exc:
        raise errors.WorkerError(f"cannot start {argv[0]!r}: {exc}") from exc
    return Process(proc)
