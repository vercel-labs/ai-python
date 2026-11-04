"""The harness tools: ``read`` and ``bash``. Both work on the local machine."""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import signal

from .. import agents
from ..types import builders, media
from ..types import messages as messages_


@agents.tool
async def read(
    path: str, offset: int | None = None, limit: int | None = None
) -> str | messages_.ContentOutput:
    """Read a file.

    offset is the 1-based first line; limit is the number of lines.
    Images (png, jpg, gif, webp) are returned as images.
    """
    p = pathlib.Path(path).expanduser()
    data = p.read_bytes()  # missing file / directory errors go to the model

    # images: return the bytes so the provider sends a real image block
    if mime := media.detect_image_media_type(data[:32]):
        return builders.content_output(
            f"Read image file [{mime}, {len(data)} bytes]",
            builders.file_part(data, media_type=mime),
        )

    lines = data.decode("utf-8", errors="replace").split("\n")
    start = max((offset or 1) - 1, 0)
    if start >= len(lines):
        raise ValueError(
            f"offset {offset} is past the end of the file "
            f"({len(lines)} lines)"
        )
    end = len(lines) if limit is None else start + limit
    return "\n".join(lines[start:end])


@agents.tool
async def bash(command: str, timeout: float = 120) -> str:
    """Run a bash command in the current directory. Returns stdout+stderr.

    timeout is in seconds.
    """
    # own process group, so we can kill the command and all its children
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    stdout = proc.stdout
    assert stdout is not None
    chunks: list[bytes] = []

    async def pump() -> None:
        # collect into `chunks` so a timeout keeps the partial output
        while chunk := await stdout.read(65536):
            chunks.append(chunk)
        await proc.wait()

    timed_out = False
    try:
        await asyncio.wait_for(pump(), timeout)
    except TimeoutError:
        timed_out = True
    finally:
        # timeout, or the run was stopped: kill the whole group. A no-op
        # if it already exited.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)

    output = b"".join(chunks).decode("utf-8", errors="replace")
    if timed_out:
        note = f"[timed out after {timeout}s]"
    elif proc.returncode:
        note = f"[exit code {proc.returncode}]"
    else:
        note = ""
    return "\n\n".join(s for s in (output, note) if s) or "[no output]"
