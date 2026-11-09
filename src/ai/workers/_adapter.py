"""Shared CLI data and executable probing; no adapter objects."""

from __future__ import annotations

import asyncio
import os
import re
import warnings
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import pydantic

from ..types import messages as messages_
from . import _errors as errors

Prompt = list[messages_.TextPart | messages_.FilePart]
Approve = Callable[[str, str, dict[str, Any]], Awaitable[tuple[bool, str]]]


class SessionInfo(pydantic.BaseModel):
    """A session in a CLI's own store. Timestamps are milliseconds."""

    session_id: str
    title: str | None = None
    cwd: str | None = None
    created_at: int | None = None
    updated_at: int | None = None


async def probe(
    kind: str,
    executable: str,
    *,
    install: str,
    verified: str,
    cwd: str,
    env: Mapping[str, str],
) -> str:
    """Check the CLI and warn if its version is older than tested."""
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
