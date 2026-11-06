"""What every adapter converts the same way.

Prompt files go in as bytes or text. The CLI's own tool calls come out as
`BuiltinTool*` events.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

from ..models.core.helpers import files
from ..types import events, media
from ..types import messages as messages_
from . import _adapter
from . import _errors as errors

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


def check_files(
    prompt: _adapter.Prompt, kind: str, accepts: Sequence[str]
) -> None:
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


def text_of(prompt: _adapter.Prompt) -> str:
    return "\n\n".join(
        p.text for p in prompt if isinstance(p, messages_.TextPart)
    )


# -- the CLI's own tools as events --------------------------------------------


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
