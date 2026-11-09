"""A CLI process: the version probe, and JSON lines on stdio.

Shared by the wires. `probe` serves every wire; `spawn` and `Process` serve
the wires that drive a CLI over stdio (Codex, Pi).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import warnings
from collections.abc import Mapping, Sequence

from . import _errors as errors

# Longest line a CLI may write. One JSON line can be megabytes (a
# `thread/read` of a long thread). With asyncio's default (64 KiB) such a line
# failed as if the CLI had crashed.
LINE_LIMIT = 256 * 1024 * 1024

# Seconds a CLI gets to exit after SIGTERM, before SIGKILL.
TERM_GRACE = 3.0


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


async def probe(
    kind: str,
    executable: str,
    *,
    install: str,
    verified: str,
    cwd: str,
    env: Mapping[str, str],
) -> str:
    """Check that the CLI runs; warn if it is older than the verified one."""
    try:
        proc = await asyncio.create_subprocess_exec(
            executable,
            "--version",
            cwd=cwd,
            env={**os.environ, **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(60):
            stdout, _ = await proc.communicate()
    except (OSError, TimeoutError) as exc:
        raise errors.ExecutableMissingError(kind, executable, install) from exc
    if proc.returncode != 0:
        raise errors.ExecutableMissingError(kind, executable, install)
    version = (stdout.decode(errors="replace").strip().splitlines() or [""])[0]
    have, want = _version(version), _version(verified)
    if have is not None and want is not None and have < want:
        warnings.warn(
            f"{kind} {version} is older than {verified}, the version this "
            "SDK was verified with; update it if something misbehaves",
            stacklevel=2,
        )
    return version


def _version(text: str | None) -> tuple[int, ...] | None:
    found = re.search(r"\d+(?:\.\d+)+", text or "")
    return tuple(int(n) for n in found.group(0).split(".")) if found else None
