"""The JSON-RPC peer codex is driven over, offline."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from ai.harnesses.experimental import errors
from ai.harnesses.experimental._adapters import jsonrpc

from . import fakes


def _peer(
    process: fakes.FakeProcess,
    on_request: Any = None,
    on_notify: Any = None,
) -> jsonrpc.JsonRpcPeer:
    async def no_notify(method: str, params: dict[str, Any]) -> None:
        pass

    async def no_request(
        request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        pass

    peer = jsonrpc.JsonRpcPeer(process, harness="test")
    peer.start(on_notify or no_notify, on_request or no_request)
    return peer


async def test_a_call_gets_its_answer() -> None:
    process = fakes.FakeProcess(lambda message: {"echo": message["method"]})
    peer = _peer(process)
    assert await peer.call("ping", timeout=1) == {"echo": "ping"}
    await peer.close()


async def test_a_call_made_while_answering_a_request_is_answered() -> None:
    """H3: answering a server request must not stop the peer reading."""
    process = fakes.FakeProcess()
    took: list[float] = []
    done = asyncio.Event()
    peer: jsonrpc.JsonRpcPeer

    async def on_request(
        request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        started = time.monotonic()
        await peer.respond(request_id, {"decision": "decline"})
        await peer.call("turn/interrupt", {}, timeout=2)
        took.append(time.monotonic() - started)
        done.set()

    peer = _peer(process, on_request=on_request)
    process.server_says({"id": 99, "method": "approval", "params": {}})
    await asyncio.wait_for(done.wait(), 3)
    assert took[0] < 0.5
    await peer.close()


async def test_notifications_flow_while_a_request_is_pending() -> None:
    """H4: a slow answer (a human approving) does not freeze the stream."""
    process = fakes.FakeProcess()
    release = asyncio.Event()
    seen: list[str] = []

    async def on_request(
        request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        await release.wait()

    async def on_notify(method: str, params: dict[str, Any]) -> None:
        seen.append(method)

    peer = _peer(process, on_request=on_request, on_notify=on_notify)
    process.server_says({"id": 1, "method": "approval", "params": {}})
    process.server_says({"method": "item/agentMessage/delta", "params": {}})
    # And a call from someone else meanwhile is answered.
    assert await peer.call("thread/read", timeout=1) == {}
    for _ in range(50):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == ["item/agentMessage/delta"]
    release.set()
    await peer.close()


async def test_notifications_keep_their_order() -> None:
    process = fakes.FakeProcess()
    seen: list[int] = []

    async def on_notify(method: str, params: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        seen.append(params["n"])

    peer = _peer(process, on_notify=on_notify)
    for n in range(20):
        process.server_says({"method": "n", "params": {"n": n}})
    await peer.call("ping", timeout=1)
    assert seen == list(range(20))
    await peer.close()


async def test_close_cancels_requests_still_being_answered() -> None:
    process = fakes.FakeProcess()
    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def on_request(
        request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    peer = _peer(process, on_request=on_request)
    process.server_says({"id": 1, "method": "approval", "params": {}})
    await asyncio.wait_for(started.wait(), 1)
    await peer.close()
    assert cancelled.is_set()


async def test_a_failing_request_handler_does_not_kill_the_peer() -> None:
    process = fakes.FakeProcess()

    async def on_request(
        request_id: Any, method: str, params: dict[str, Any]
    ) -> None:
        raise ValueError("bug in the handler")

    peer = _peer(process, on_request=on_request)
    process.server_says({"id": 1, "method": "approval", "params": {}})
    assert await peer.call("ping", timeout=1) == {}
    await peer.close()
    with pytest.raises(errors.AgentCrashedError):
        await peer.call("ping", timeout=1)
