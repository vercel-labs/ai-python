"""The private contract between `Worker` and one CLI.

A wire speaks one CLI's native protocol and nothing else. It turns native
records into `ai` events. It keeps no conversation state, knows no
subscribers, and does not know about `ai.Agent`: the worker does all that.

A wire yields three kinds of items:

- `Content`: `ai` model events for what the assistant says and does. The CLI's
  own tools are `BuiltinTool*` events. Calls to the MCP bridge (`ai` tools)
  must NOT appear: the bridge reports them to the worker itself.
- `Accepted`: the CLI took a message from `send()`. It must come before any
  content that the message causes.
- `TurnEnd`: the CLI finished a turn.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import re
import warnings
from collections.abc import AsyncIterator, Mapping
from typing import Literal, Protocol

from ..types import events
from ..types import messages as messages_
from ..types import usage as usage_
from . import _errors as errors

Content = (
    events.TextStart
    | events.TextDelta
    | events.TextEnd
    | events.ReasoningStart
    | events.ReasoningDelta
    | events.ReasoningEnd
    | events.BuiltinToolStart
    | events.BuiltinToolDelta
    | events.BuiltinToolEnd
    | events.BuiltinToolResult
)


@dataclasses.dataclass(frozen=True)
class Accepted:
    """The CLI took this message: it starts a turn or steers one."""

    message: messages_.Message


@dataclasses.dataclass(frozen=True)
class TurnEnd:
    """The CLI finished a turn.

    `error` is set when the turn failed; the worker raises it from `stream()`.
    """

    reason: Literal["stop", "length", "interrupted", "error"] = "stop"
    usage: usage_.Usage | None = None
    error: errors.WorkerError | None = None


Item = Content | Accepted | TurnEnd


class Wire(Protocol):
    """One CLI process, driven by its native protocol."""

    async def start(
        self,
        *,
        bridge_url: str,
        instructions: str | None,
        seed: list[messages_.Message],
    ) -> str:
        """Start the CLI and open the session; return the session id.

        Resume `worker.session_id` if it is set. Otherwise, start a new
        session and give it `seed` as its history (lossy).
        """
        ...

    async def send(self, message: messages_.Message) -> None:
        """Give a user message to the CLI: a new turn, or a steer."""
        ...

    def events(self) -> AsyncIterator[Item]:
        """Yield the items of the whole session. One reader only."""
        ...

    async def interrupt(self) -> None:
        """Ask the CLI to stop the current turn. Do not wait for it."""
        ...

    async def close(self) -> None:
        """Stop the CLI. Safe to call more than once."""
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
