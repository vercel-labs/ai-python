"""The codex adapter against a scripted app-server, offline."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from ai.harnesses.experimental import Allow, ApprovalContext, Deny
from ai.harnesses.experimental._adapters import codex, jsonrpc
from ai.harnesses.experimental.errors import UnsupportedError
from ai.types import events
from ai.types import messages as m

from . import fakes

THREAD = "019a0000-0000-7000-8000-000000000000"


class AppServer:
    """Answers like `codex app-server` for one turn that asks to run a
    command, and ends the turn when it is interrupted."""

    def __init__(self) -> None:
        self.process = fakes.FakeProcess(self.answer)
        self.interrupted = False

    def answer(self, message: dict[str, Any]) -> Any:
        method = message.get("method")
        if method == "turn/start":
            self.process.server_says(
                {"method": "turn/started", "params": {"threadId": THREAD}}
            )
            self.process.server_says(
                {
                    "id": 500,
                    "method": "item/commandExecution/requestApproval",
                    "params": {
                        "threadId": THREAD,
                        "turnId": "T1",
                        "itemId": "i1",
                        "command": "rm -rf /",
                    },
                }
            )
            return {"turn": {"id": "T1"}}
        if method == "turn/interrupt":
            self.interrupted = True
            self.process.server_says(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": THREAD,
                        "turn": {"id": "T1", "status": "interrupted"},
                    },
                }
            )
            return {}
        if "method" in message and "id" in message:
            return {}
        return None


def _adapter(server: Any, approve: Any) -> codex.CodexAdapter:
    adapter = codex.CodexAdapter()
    adapter.approve = approve
    peer = jsonrpc.JsonRpcPeer(server.process, harness="codex")
    adapter._workspace = fakes.FakeWorkspace()
    adapter._process = server.process
    adapter._peer = peer
    peer.start(
        adapter._on_notify, adapter._on_request, on_crash=adapter._on_crash
    )
    return adapter


async def _close(adapter: codex.CodexAdapter) -> None:
    if adapter._peer is not None:
        await adapter._peer.close()


async def test_deny_and_stop_ends_the_turn_at_once() -> None:
    """H3: measured 30s before, waiting on a reply the reader could not read."""
    server = AppServer()

    async def deny(ctx: ApprovalContext) -> Deny:
        return Deny("no", stop=True)

    adapter = _adapter(server, deny)
    started = time.monotonic()
    seen: list[Any] = []
    async with asyncio.timeout(5):
        async for event in adapter.turn(THREAD, [m.TextPart(text="clean up")]):
            seen.append(event)
    assert time.monotonic() - started < 1
    assert server.interrupted
    assert isinstance(seen[-1], events.StreamEnd)
    assert seen[-1].finish_reason == "cancelled"
    declined = [
        m
        for m in server.process.written
        if m.get("id") == 500 and "result" in m
    ]
    assert declined == [
        {"jsonrpc": "2.0", "id": 500, "result": {"decision": "decline"}}
    ]
    await _close(adapter)


async def test_a_slow_approval_does_not_block_other_calls() -> None:
    """H4: while a person decides, the rest of the connection still works."""
    server = AppServer()
    deciding = asyncio.Event()
    decide = asyncio.Event()

    async def slow(ctx: ApprovalContext) -> Deny:
        deciding.set()
        await decide.wait()
        return Deny("no", stop=True)

    adapter = _adapter(server, slow)
    turn = adapter.turn(THREAD, [m.TextPart(text="clean up")])
    reader = asyncio.ensure_future(_drain(turn))
    await asyncio.wait_for(deciding.wait(), 2)
    # A call meanwhile, e.g. an approval hook reading the history.
    assert await adapter._rpc().call("thread/read", {}, timeout=1) == {}
    decide.set()
    await asyncio.wait_for(reader, 2)
    await _close(adapter)


async def _drain(turn: Any) -> None:
    async for _ in turn:
        pass


class TwoThreads:
    """An app-server that starts a turn per thread and leaves the rest to
    the test: it pushes approvals and completions by hand."""

    def __init__(self) -> None:
        self.process = fakes.FakeProcess(self.answer)

    def answer(self, message: dict[str, Any]) -> Any:
        if message.get("method") == "turn/start":
            return {"turn": {"id": "T-" + message["params"]["threadId"]}}
        if "method" in message and "id" in message:
            return {}
        return None

    def complete(self, thread: str) -> None:
        self.process.server_says(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": thread,
                    "turn": {"id": "T-" + thread, "status": "completed"},
                },
            }
        )


async def test_an_unsupported_decision_fails_only_its_own_turn() -> None:
    """H7: one parked error per adapter, cleared by whichever turn started
    next, raised by whichever turn ended first."""
    server = TwoThreads()

    async def approve(ctx: ApprovalContext) -> Allow:
        return Allow(input={"command": "ls"})  # codex cannot rewrite

    adapter = _adapter(server, approve)
    a = adapter.turn("A", [m.TextPart(text="one")])
    assert isinstance(await anext(a), events.StreamStart)
    await adapter._decide(
        "item/commandExecution/requestApproval",
        {"threadId": "A", "turnId": "T-A", "itemId": "i1"},
    )
    b = adapter.turn("B", [m.TextPart(text="two")])
    assert isinstance(await anext(b), events.StreamStart)  # B starts after
    server.complete("B")
    ended = [event async for event in b]
    assert isinstance(ended[-1], events.StreamEnd), "B is not A's failure"
    server.complete("A")
    with pytest.raises(UnsupportedError):
        async for _ in a:
            pass
    await _close(adapter)
