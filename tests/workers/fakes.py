"""Test doubles for workers: a scripted wire and an in-memory process."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import urllib.parse
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any, Literal

import pydantic

from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.workers import _convert, _wire


@dataclasses.dataclass
class CallTool:
    """The CLI calls an `ai` tool through the MCP bridge, over HTTP.

    `then(text, is_error)` gives what the CLI says after the result.
    """

    name: str
    arguments: dict[str, Any] = dataclasses.field(default_factory=dict)
    then: Callable[[str, bool], Iterable[_wire.Item]] = lambda text_, err: [
        *text(f"got {text_}", block=f"after-{text_}"),
        _wire.TurnEnd(),
    ]


Step = _wire.Item | CallTool
Reply = Callable[[m.Message], Iterable[Step]]


def text(chunk: str, block: str = "t") -> list[_wire.Item]:
    return [
        events.TextStart(block_id=block),
        events.TextDelta(chunk=chunk, block_id=block),
        events.TextEnd(block_id=block),
    ]


def builtin(
    call_id: str, name: str, args: str = "{}", result: Any = "ok"
) -> list[_wire.Item]:
    """One of the CLI's own tool calls, from start to result."""
    return [
        *_convert.builtin_call(call_id, name, args, started=set()),
        _convert.builtin_result(call_id, name, result, is_error=False),
    ]


def echo(message: m.Message) -> list[_wire.Item]:
    """Accept the message, answer with its text, end the turn."""
    return [
        _wire.Accepted(message),
        *text(message.text, block=f"t-{message.id}"),
        _wire.TurnEnd(),
    ]


class FakeWire:
    """A CLI that says what the test tells it to.

    `reply(message)` gives the items that a sent message causes (default:
    accept it, nothing more). `push()` adds items at any time.
    """

    def __init__(self, worker: FakeWorker) -> None:
        self.worker = worker
        self.reply: Reply = worker._reply or (lambda msg: [_wire.Accepted(msg)])
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.started: dict[str, Any] | None = None
        self.sent: list[m.Message] = []
        self.interrupts = 0
        self.calls: list[asyncio.Task[tuple[str, bool]]] = []
        self.closed = False
        # What `interrupt()` does. None: the CLI ignores interrupts.
        self.on_interrupt: list[_wire.Item] | None = [
            _wire.TurnEnd(reason="interrupted")
        ]
        worker._wires.append(self)

    def push(self, *items: Step | BaseException) -> None:
        for item in items:
            self.queue.put_nowait(item)

    async def start(
        self,
        *,
        session_id: str | None,
        bridge_url: str,
        instructions: str | None,
        seed: list[m.Message],
    ) -> str:
        self.started = {
            "bridge_url": bridge_url,
            "instructions": instructions,
            "seed": seed,
            "session_id": session_id,
        }
        return session_id or "s1"

    async def send(self, message: m.Message) -> None:
        self.sent.append(message)
        self.push(*self.reply(message))

    async def events(self) -> AsyncIterator[_wire.Item]:
        while True:
            item = await self.queue.get()
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, CallTool):
                task = asyncio.create_task(self._call(item))
                self.calls.append(task)
                continue
            yield item

    async def _call(self, call: CallTool) -> tuple[str, bool]:
        assert self.started is not None
        reply = await mcp_call(
            self.started["bridge_url"],
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": call.name, "arguments": call.arguments},
            },
        )
        result = reply["result"]
        answer = (result["content"][0]["text"], bool(result["isError"]))
        self.push(*call.then(*answer))
        return answer

    async def interrupt(self) -> None:
        self.interrupts += 1
        if self.on_interrupt is not None:
            self.push(*self.on_interrupt)

    async def close(self) -> None:
        self.closed = True


class FakeProvider(workers.WorkerProvider):
    name: str = "fake"

    def _make_wire(self, worker: FakeWorker) -> FakeWire:
        return FakeWire(worker)


class FakeWorker(workers.Worker):
    worker_class_id: Literal["fake"] = "fake"
    provider: FakeProvider = pydantic.Field(default_factory=FakeProvider)

    # Test controls, not settings.
    _reply: Reply | None = pydantic.PrivateAttr(default=None)
    _wires: list[FakeWire] = pydantic.PrivateAttr(default_factory=list)

    @property
    def wire(self) -> FakeWire:
        return self._wires[-1]


def fake(reply: Reply | None = None, **fields: Any) -> FakeWorker:
    worker = FakeWorker(**fields)
    worker._reply = reply
    return worker


class FakeProcess:
    """In-memory duplex stdio, for the JSON-RPC peer and the wires.

    `answer(message)` is called for every message the client writes. For a
    request, a non-None return is sent back as its result. `server_says()`
    pushes a line from the "server": a notification or a request.
    """

    def __init__(
        self, answer: Callable[[dict[str, Any]], Any] | None = None
    ) -> None:
        self._lines: asyncio.Queue[str] = asyncio.Queue()
        self.answer = answer or (lambda message: {})
        self.written: list[dict[str, Any]] = []
        self.terminated = False

    def server_says(self, message: dict[str, Any]) -> None:
        self._lines.put_nowait(json.dumps(message) + "\n")

    async def write(self, data: str) -> None:
        message = json.loads(data)
        self.written.append(message)
        result = self.answer(message)
        if "method" in message and "id" in message and result is not None:
            self.server_says({"id": message["id"], "result": result})

    async def readline(self) -> str:
        return await self._lines.get()

    async def read_stderr(self) -> str:
        return ""

    async def wait(self) -> int:
        return 0

    async def terminate(self) -> None:
        self.terminated = True
        self._lines.put_nowait("")

    async def death_note(self) -> str:
        return "exit code 0; no stderr"


async def mcp_call(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST one JSON-RPC message to the bridge; return the JSON reply."""
    parts = urllib.parse.urlsplit(url)
    reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
    body = json.dumps(payload).encode()
    writer.write(
        f"POST {parts.path} HTTP/1.1\r\nHost: x\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        "\r\n".encode()
        + body
    )
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    length = 0
    for line in head.decode().split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1])
    data = await reader.readexactly(length)
    writer.close()
    return json.loads(data) if data else {}
