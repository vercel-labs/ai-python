"""Test doubles for workers: a scripted adapter and an in-memory process."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any, Literal

import pydantic

from ai import workers
from ai.types import events
from ai.types import messages as m
from ai.types import usage as usage_
from ai.workers import _adapter, _convert


@dataclasses.dataclass
class Pause:
    """Wait here until the turn is interrupted (or cancelled)."""


@dataclasses.dataclass
class Run:
    """Await a coroutine here, mid-turn: a bridge call, say."""

    fn: Callable[[], Awaitable[None]]


@dataclasses.dataclass
class Ask:
    """Ask the worker to approve one of the CLI's own tool calls."""

    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any] = dataclasses.field(default_factory=dict)


Step = events.Event | Pause | Run | Ask
Script = Callable[[str], Iterable[Step]]


def text(chunk: str, block: str = "t") -> list[Step]:
    return [
        events.TextStart(block_id=block),
        events.TextDelta(chunk=chunk, block_id=block),
        events.TextEnd(block_id=block),
    ]


def tool(
    call_id: str, name: str, args: str = "{}", result: Any = "ok"
) -> list[Step]:
    """One of the CLI's own tool calls, start to result."""
    return [
        *_convert.builtin_call(call_id, name, args, started=set()),
        _convert.builtin_result(call_id, name, result, is_error=False),
    ]


def reply(prompt: str) -> list[Step]:
    """Echo the prompt back."""
    return text(prompt)


class FakeAdapter:
    def __init__(
        self, script: Script, approve: _adapter.Approve | None
    ) -> None:
        self.script = script
        self.approve = approve
        self.version: str | None = None
        self.opens: list[dict[str, Any]] = []
        self.turns: list[str] = []
        self.steers: list[str] = []
        self.interrupts = 0
        self.answers: list[tuple[bool, str]] = []
        self.closed = False
        self.stored: dict[str, list[m.Message]] = {}
        self._resume = asyncio.Event()

    async def start(self) -> None:
        self.version = "fake 1.0"

    async def close(self) -> None:
        self.closed = True

    async def open(
        self,
        *,
        session_id: str | None,
        fork: bool,
        seed: list[m.Message] | None,
        instructions: str | None,
    ) -> str:
        self.opens.append(
            {
                "session_id": session_id,
                "fork": fork,
                "seed": seed,
                "instructions": instructions,
            }
        )
        if fork:
            return f"{session_id}-fork"
        return session_id or "s1"

    async def turn(
        self, prompt: _adapter.Prompt
    ) -> AsyncIterator[events.Event]:
        self.turns.append(_convert.text_of(prompt))
        self._resume.clear()
        yield events.StreamStart()
        for step in self.script(_convert.text_of(prompt)):
            if isinstance(step, Pause):
                await self._resume.wait()
                yield events.StreamEnd(finish_reason="cancelled")
                return
            if isinstance(step, Run):
                await step.fn()
            elif isinstance(step, Ask):
                assert self.approve is not None
                self.answers.append(
                    await self.approve(
                        step.tool_call_id, step.tool_name, step.tool_input
                    )
                )
            else:
                yield step
        yield events.StreamEnd(
            finish_reason="stop",
            usage=usage_.Usage(input_tokens=3, output_tokens=2),
        )

    async def steer(self, text: str) -> None:
        self.steers.append(text)

    async def interrupt(self) -> None:
        self.interrupts += 1
        self._resume.set()

    async def history(self, session_id: str) -> list[m.Message]:
        return self.stored.get(session_id, [])

    async def list_sessions(self) -> list[_adapter.SessionInfo]:
        return [_adapter.SessionInfo(session_id="s1", title="first")]

    def tui_command(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        return (
            [sys.executable, "-c", "pass"],
            {"FAKE_TUI": "1"},
            session_id or "minted",
        )


class FakeWorker(workers.Worker):
    """A worker whose CLI is a `FakeAdapter`, run by a script."""

    kind: Literal["fake"] = "fake"
    _script: Script = pydantic.PrivateAttr(default_factory=lambda: reply)
    _fake: FakeAdapter | None = pydantic.PrivateAttr(default=None)
    _servers: dict[str, dict[str, Any]] = pydantic.PrivateAttr(
        default_factory=dict
    )

    def _make_adapter(
        self,
        mcp_servers: dict[str, dict[str, Any]],
        approve: _adapter.Approve | None,
    ) -> _adapter.CLIAdapter:
        self._fake = FakeAdapter(self._run, approve)
        self._servers = mcp_servers
        return self._fake

    def _run(self, prompt: str) -> Iterable[Step]:
        return self._script(prompt)

    @property
    def servers(self) -> dict[str, dict[str, Any]]:
        return self._servers

    def script(self, script: Script) -> None:
        self._script = script

    @property
    def fake(self) -> FakeAdapter:
        assert self._fake is not None
        return self._fake


class FakeProcess:
    """In-memory duplex stdio, for the JSON-RPC peer and the adapters.

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
        self._lines.put_nowait("")

    async def death_note(self) -> str:
        return "exit code 0; no stderr"
