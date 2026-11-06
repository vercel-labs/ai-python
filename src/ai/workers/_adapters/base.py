"""What every adapter shares: its protocol, local processes, prompt files.

An adapter drives ONE conversation with one CLI. It speaks the model view
both ways: `turn()` yields the events of one assistant message, with the
CLI's own tools as `BuiltinTool*` events, and ends with a `StreamEnd`; a
history it reads or is seeded with is a plain `list[Message]`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import re
import warnings
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from typing import Any, Protocol

import pydantic

from ...models.core.helpers import files
from ...types import events, media
from ...types import messages as messages_
from .. import _errors as errors

Prompt = list[messages_.TextPart | messages_.FilePart]

Approve = Callable[[str, str, dict[str, Any]], Awaitable[tuple[bool, str]]]
"""`approve(tool_call_id, tool_name, tool_input)` returns `(granted, reason)`.

Never raises: the worker answers a failure with a denial."""

LINE_LIMIT = 256 * 1024 * 1024
"""Longest line a CLI may write. The protocols are JSON lines, and one line
can be megabytes (a codex `thread/read` of a long thread). With asyncio's
default (64 KiB) such a line failed as if the CLI had crashed."""

TERM_GRACE = 3.0
"""Seconds a CLI gets to exit after SIGTERM, before SIGKILL."""


class SessionInfo(pydantic.BaseModel):
    """A conversation in a CLI's own store."""

    session_id: str
    title: str | None = None
    cwd: str | None = None
    created_at: int | None = None
    """Milliseconds since the epoch, when the CLI records it."""
    updated_at: int | None = None


class Adapter(Protocol):
    version: str | None
    """What `<cli> --version` printed, once started."""

    async def start(self) -> None:
        """Check the CLI runs, and start what it needs (codex: a server)."""
        ...

    async def close(self) -> None: ...

    async def open(
        self,
        *,
        session_id: str | None,
        fork: bool,
        seed: list[messages_.Message] | None,
        instructions: str | None,
    ) -> str:
        """Open the conversation and return its session id.

        A new one (seeded with a past when `seed` is given), the stored one
        `session_id` names, or a native fork of it.
        """
        ...

    def turn(self, prompt: Prompt) -> AsyncIterator[events.Event]:
        """Send a prompt; yield one assistant message, ending in StreamEnd."""
        ...

    async def steer(self, text: str) -> None:
        """Send guidance to the turn that runs now."""
        ...

    async def interrupt(self) -> None:
        """End the running turn. The CLI keeps its transcript."""
        ...

    async def history(self, session_id: str) -> list[messages_.Message]:
        """Read a stored conversation."""
        ...

    async def list_sessions(self) -> list[SessionInfo]: ...

    def tui_command(
        self, session_id: str | None
    ) -> tuple[list[str], dict[str, str], str | None]:
        """Return the CLI's interactive command, its env, and its session id.

        The session id is minted here for a new conversation when the CLI
        lets us choose it, else None.
        """
        ...


# -- local processes -----------------------------------------------------------


class Process:
    """A CLI started by `spawn`: JSON lines on stdin and stdout."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return self._proc.pid

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
    """Check that the CLI runs, and return its version.

    Warns when it is older than the version the adapter was verified with:
    the adapters lean on CLI behavior that is measured, not documented.
    """
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


# -- files in a prompt ---------------------------------------------------------
#
# Files are turned into bytes HERE: a URL is downloaded with the SDK's own
# SSRF-checked downloader, not handed to the CLI. Measured: Claude Code drops
# an image given by URL without a word.


async def data(part: messages_.FilePart) -> bytes:
    """Return a file's bytes, downloading a URL first."""
    if isinstance(part.data, bytes):
        return part.data
    if media.is_downloadable_url(part.data):
        body, _ = await files.download(part.data)
        return body
    return base64.b64decode(media.data_to_base64(part.data))


async def as_text(part: messages_.FilePart) -> str:
    """Return a text file as text a CLI reads inline, named if it has one."""
    body = (await data(part)).decode("utf-8", errors="replace")
    name = part.filename or "attached file"
    return f"<file name={name!r}>\n{body}\n</file>"


def check_files(prompt: Prompt, kind: str, accepts: Sequence[str]) -> None:
    """Refuse a file the CLI cannot take, before anything runs.

    Text files (`text/*`) go everywhere, as text. `accepts` lists the other
    media type prefixes the CLI takes.
    """
    for part in prompt:
        if isinstance(part, messages_.FilePart) and not any(
            part.media_type.startswith(p) for p in ("text/", *accepts)
        ):
            raise errors.WorkerError(
                f"{kind} does not take {part.media_type} files"
            )


def text_of(prompt: Prompt) -> str:
    return "\n\n".join(
        p.text for p in prompt if isinstance(p, messages_.TextPart)
    )


def builtin_call(
    tool_call_id: str,
    tool_name: str,
    tool_args: str,
    *,
    started: set[str],
) -> list[events.Event]:
    """Return the events of one finished CLI tool call.

    `started` holds the calls whose start was already sent: the start is
    sent here otherwise, because the hydrator only completes a call it saw
    start. The arguments travel as a delta: the hydrator builds a call's
    arguments only from its deltas.
    """
    out: list[events.Event] = []
    if tool_call_id not in started:
        started.add(tool_call_id)
        out.append(
            events.BuiltinToolStart(
                tool_call_id=tool_call_id, tool_name=tool_name
            )
        )
    return [
        *out,
        events.BuiltinToolDelta(tool_call_id=tool_call_id, chunk=tool_args),
        events.BuiltinToolEnd(
            tool_call_id=tool_call_id,
            tool_call=messages_.BuiltinToolCallPart(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_args=tool_args,
            ),
        ),
    ]


def builtin_result(
    tool_call_id: str, tool_name: str, result: Any, *, is_error: bool
) -> events.BuiltinToolResult:
    return events.BuiltinToolResult(
        tool_call_id=tool_call_id,
        result=messages_.BuiltinToolReturnPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
            is_error=is_error,
        ),
    )
