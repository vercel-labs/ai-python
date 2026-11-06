"""The MCP bridge: `ai` tools, served to a coding-agent CLI.

A worker's CLI runs its own loop and its own tools. To give it the tools an
`ai.Agent` holds, the worker serves them over MCP on localhost, and the CLI
connects to that server like any other.

The server does not run a tool. It hands each call to `on_call` and waits
for the answer. The worker turns the call into a normal `ToolCallPart`, ends
its stream with `finish_reason="tool_call"`, and `ai.Agent` runs the tool.
The next `stream()` call brings the result, and the worker answers the
waiting request with it. While the request waits, the CLI waits too.

Only what an MCP client needs is implemented: streamable HTTP, POST only,
plain JSON responses (no SSE), and the methods `initialize`, `ping`,
`tools/list` and `tools/call`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import tools as tools_

SERVER_NAME = "ai_sdk"
"""The MCP server name the CLI sees. Claude Code calls a tool
`mcp__ai_sdk__<name>`."""

PROTOCOL_VERSION = "2025-06-18"

OnCall = Callable[[str, dict[str, Any]], Awaitable[tuple[str, bool]]]
"""`on_call(name, arguments)` returns `(text, is_error)`."""


class Bridge:
    def __init__(self, on_call: OnCall) -> None:
        self._on_call = on_call
        self._tools: dict[str, tools_.Tool] = {}
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("bridge is not started")
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"http://{host}:{port}/mcp"

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def set_tools(self, tools: list[tools_.Tool]) -> None:
        """Serve these tools from now on.

        A CLI reads the list when it connects, so a tool added after that
        may stay unseen until the next connection.
        """
        self._tools = {
            t.name: t for t in tools if t.kind == "function" and t.spec
        }

    async def start(self) -> None:
        if self._server is None:
            self._server = await asyncio.start_server(
                self._serve, host="127.0.0.1", port=0
            )

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        for task in list(self._connections):
            task.cancel()
        for task in list(self._connections):
            with contextlib.suppress(BaseException):
                await task
        with contextlib.suppress(Exception):
            await self._server.wait_closed()
        self._server = None

    # -- HTTP -----------------------------------------------------------------

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._connections.add(task)
        try:
            while True:
                request = await _read_request(reader)
                if request is None:
                    return
                method, body = request
                if method != "POST":
                    # No SSE stream to offer, and a session needs no teardown.
                    status = 200 if method == "DELETE" else 405
                    await _respond(writer, status, None)
                    continue
                reply = await self._dispatch(body)
                await _respond(writer, 200 if reply is not None else 202, reply)
        except (ConnectionError, asyncio.IncompleteReadError):
            return
        finally:
            self._connections.discard(task)
            with contextlib.suppress(Exception):
                writer.close()

    # -- JSON-RPC -------------------------------------------------------------

    async def _dispatch(self, body: bytes) -> Any:
        try:
            payload = json.loads(body or b"null")
        except json.JSONDecodeError:
            return _error(None, -32700, "parse error")
        if isinstance(payload, list):
            replies = [await self._one(item) for item in payload]
            kept = [r for r in replies if r is not None]
            return kept or None
        return await self._one(payload)

    async def _one(self, message: Any) -> Any:
        if not isinstance(message, dict) or "method" not in message:
            # A response from the client (we send no requests) or junk.
            return None
        method, rid = message["method"], message.get("id")
        if rid is None:
            return None  # a notification: nothing to answer
        params = message.get("params") or {}
        if method == "initialize":
            return _result(
                rid,
                {
                    "protocolVersion": params.get(
                        "protocolVersion", PROTOCOL_VERSION
                    ),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": "0"},
                },
            )
        if method == "ping":
            return _result(rid, {})
        if method == "tools/list":
            return _result(
                rid,
                {
                    "tools": [
                        {
                            "name": t.name,
                            "description": (t.spec and t.spec.description)
                            or "",
                            "inputSchema": (t.spec and t.spec.params)
                            or {"type": "object"},
                        }
                        for t in self._tools.values()
                    ]
                },
            )
        if method == "tools/call":
            name = str(params.get("name", ""))
            if name not in self._tools:
                return _error(rid, -32602, f"unknown tool: {name}")
            arguments = params.get("arguments") or {}
            text, is_error = await self._on_call(name, dict(arguments))
            return _result(
                rid,
                {
                    "content": [{"type": "text", "text": text}],
                    "isError": is_error,
                },
            )
        return _error(rid, -32601, f"method not found: {method}")


def _result(rid: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": code, "message": message},
    }


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, bytes] | None:
    """Read one HTTP/1.1 request. None when the client hung up."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError:
        return None
    lines = head.decode("latin-1").split("\r\n")
    method = lines[0].split(" ", 1)[0].upper()
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    if headers.get("transfer-encoding", "").lower() == "chunked":
        body = b""
        while True:
            size = int((await reader.readuntil(b"\r\n")).strip(), 16)
            if size == 0:
                await reader.readuntil(b"\r\n")
                break
            body += await reader.readexactly(size)
            await reader.readexactly(2)
        return method, body
    length = int(headers.get("content-length", "0") or 0)
    return method, await reader.readexactly(length) if length else b""


async def _respond(
    writer: asyncio.StreamWriter, status: int, payload: Any
) -> None:
    reason = {200: "OK", 202: "Accepted", 405: "Method Not Allowed"}[status]
    body = b"" if payload is None else json.dumps(payload).encode()
    head = [f"HTTP/1.1 {status} {reason}", f"Content-Length: {len(body)}"]
    if payload is not None:
        head.append("Content-Type: application/json")
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + body)
    await writer.drain()
