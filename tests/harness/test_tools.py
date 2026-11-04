from __future__ import annotations

import asyncio
import os
import pathlib
import time

import pytest

from ai.harness import tools
from ai.types import messages as messages_

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
)


async def test_read_window(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("\n".join(f"line {i}" for i in range(1, 11)))

    assert await tools.read.fn(str(f), offset=3, limit=2) == "line 3\nline 4"
    assert await tools.read.fn(str(f), offset=10) == "line 10"
    with pytest.raises(ValueError, match="past the end"):
        await tools.read.fn(str(f), offset=11)


async def test_read_returns_the_whole_file(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "big.txt"
    text = "\n".join(str(i) for i in range(5000))
    f.write_text(text)

    assert await tools.read.fn(str(f)) == text


async def test_read_image(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "a.png"
    f.write_bytes(PNG)

    out = await tools.read.fn(str(f))

    assert isinstance(out, messages_.ContentOutput)
    assert isinstance(out.value[1], messages_.FilePart)
    assert out.value[1].media_type == "image/png"


async def test_bash_output_and_exit_code() -> None:
    assert await tools.bash.fn("echo hi; echo err >&2") == "hi\nerr\n"
    assert await tools.bash.fn("echo no; exit 3") == "no\n\n\n[exit code 3]"
    assert await tools.bash.fn("exit 3") == "[exit code 3]"
    assert await tools.bash.fn("true") == "[no output]"


async def test_bash_returns_the_whole_output() -> None:
    out = await tools.bash.fn("seq 1 5000")
    assert out == "".join(f"{i}\n" for i in range(1, 5001))


async def test_bash_timeout_keeps_partial_output() -> None:
    out = await tools.bash.fn("echo started; sleep 10", timeout=0.3)
    assert out == "started\n\n\n[timed out after 0.3s]"


async def test_bash_cancel_kills_process_group(tmp_path: pathlib.Path) -> None:
    pidfile = tmp_path / "pid"
    task = asyncio.create_task(
        tools.bash.fn(f"sleep 30 & echo $! > {pidfile}; wait")
    )
    while not pidfile.exists() or not pidfile.read_text().strip():
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    child = int(pidfile.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.01)
    pytest.fail("the background child survived cancellation")
