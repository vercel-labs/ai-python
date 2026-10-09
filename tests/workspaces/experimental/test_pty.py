"""`FramedPty` over a channel that refuses a second writer mid-send."""

from __future__ import annotations

import asyncio

from ai.workspaces.experimental import _pty


async def test_sends_from_many_tasks_go_one_at_a_time_in_order() -> None:
    """The sandbox's interactive stream raises BusyResourceError on a second
    send while one is in flight; the bridge sends a keystroke per task, so
    typing fast on a slow link once ended the bridge."""
    sent: list[bytes] = []
    busy = False

    async def send_bytes(data: bytes) -> None:
        nonlocal busy
        if busy:
            raise RuntimeError("BusyResourceError: another task is sending")
        busy = True
        await asyncio.sleep(0.01)  # a slow link
        sent.append(data)
        busy = False

    async def recv_bytes() -> bytes:
        await asyncio.Event().wait()
        return b""

    async def close_channel() -> None:
        pass

    pty = _pty.FramedPty(
        name=None,
        send_bytes=send_bytes,
        recv_bytes=recv_bytes,
        close_channel=close_channel,
    )
    keys = [bytes([c]) for c in b"hello"]
    await asyncio.gather(*(pty.send(k) for k in keys), pty.resize(80, 24))
    assert sent[:5] == [_pty.frame(_pty.INPUT, k) for k in keys]
    assert sent[5] == _pty.frame(_pty.RESIZE, b"80,24")
    await pty.detach()
