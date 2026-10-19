"""`FramedPty` over a channel that refuses a second writer mid-send."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

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


async def _gone_soon(pid: int, within: float) -> bool:
    import os

    loop = asyncio.get_running_loop()
    end = loop.time() + within
    while loop.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.1)
    return False


async def test_close_ends_a_program_that_ignores_the_hangup(
    tmp_path: Any,
) -> None:
    """`close()` hung up the pty and nothing else: a program that ignores
    SIGHUP ran on forever."""
    import os
    import sys

    from ai.workspaces.experimental import Local

    script = (
        "import signal, time; signal.signal(signal.SIGHUP, signal.SIG_IGN); "
        "print('up', flush=True); time.sleep(120)"
    )
    async with Local(tmp_path) as ws:
        pty = await ws.pty([sys.executable, "-c", script])
        pid = pty.pid
        assert pid is not None
        await pty.close()
        assert await _gone_soon(pid, 10), "still running after close()"
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, 9)


async def test_a_program_started_under_nohup_still_ends_on_close(
    tmp_path: Any,
) -> None:
    """Measured: under `nohup` (SIGHUP ignored, which children inherit
    across exec) every TUI a test opened outlived the test — two codex TUIs
    per local e2e run, alive for days."""
    import os
    import signal

    from ai.workspaces.experimental import Local

    before = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    pid: int | None = None
    try:
        async with Local(tmp_path) as ws:
            pty = await ws.pty(["sleep", "120"])
            pid = pty.pid
            assert pid is not None
            await pty.close()
            assert await _gone_soon(pid, 3), "the hangup must reach it"
    finally:
        signal.signal(signal.SIGHUP, before)
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 9)


async def test_an_unnamed_pty_leaves_no_directory_behind(
    monkeypatch: Any,
) -> None:
    """Each unnamed pty's socket lives in a private temp directory; the
    socket was removed when it ended, the directory never was (measured:
    239 empty harness-pty-* directories)."""
    import shutil
    import tempfile
    from pathlib import Path

    from ai.workspaces.experimental import Local

    # Short: a Unix socket path is capped at 104 bytes on macOS, and pytest's
    # own temp path is longer than that.
    temp = Path(tempfile.mkdtemp(dir="/tmp", prefix="ptyt-"))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    try:
        async with Local(temp.parent) as ws:
            pty = await ws.pty(["sh", "-c", "echo hi"])
            await pty.wait()
            await pty.close()
        for _ in range(50):
            if not list(temp.iterdir()):
                break
            await asyncio.sleep(0.1)
        assert list(temp.iterdir()) == []
    finally:
        shutil.rmtree(temp, ignore_errors=True)
