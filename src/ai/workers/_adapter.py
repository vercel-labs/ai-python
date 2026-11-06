"""The adapter protocol: what `Worker` needs from each CLI.

An adapter drives ONE conversation with one CLI. Outside it, the CLI's own
tools are built-in tools, and nothing else: `turn()` yields them as
`BuiltinTool*` events, and a history it reads holds them as built-in parts,
each result in the assistant message of its call. A seed holds them the
same way, next to the calls and `tool` messages of the tools an `ai.Agent`
ran. How the CLI stores a tool call is the adapter's business alone.

`turn()` yields the events of one assistant message and ends with a
`StreamEnd`.
"""

from __future__ import annotations

import asyncio
import os
import re
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol

import pydantic

from ..types import events
from ..types import messages as messages_
from . import _errors as errors

Prompt = list[messages_.TextPart | messages_.FilePart]

# `approve(tool_call_id, tool_name, tool_input)` returns `(granted, reason)`.
#
# Never raises: the worker answers a failure with a denial.
Approve = Callable[[str, str, dict[str, Any]], Awaitable[tuple[bool, str]]]


class SessionInfo(pydantic.BaseModel):
    """A conversation in a CLI's own store."""

    session_id: str
    title: str | None = None
    cwd: str | None = None
    created_at: int | None = None
    """Milliseconds since the epoch, when the CLI records it."""
    updated_at: int | None = None


class CLIAdapter(Protocol):
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
