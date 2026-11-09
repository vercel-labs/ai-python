"""Conversions that every wire does the same way. Only wires use them.

Prompt files go in as bytes or text. The CLI's own tool calls come out as
`BuiltinTool*` events. A past seeds a new CLI session by one rule:

A worker given `[*past, user_message]` on its first turn starts a new
conversation from `past`. Each CLI stores it natively so the agent truly
has the context, and the rule for what becomes what is the same everywhere:

- text: verbatim, same role.
- tool calls and results: the NATIVE tool items, original name and
  arguments unaltered. Measured: the receiving agent reads them as history
  and acts with its own tools; it does not try to call a tool it lacks.
  Names are not translated — mapping `Write` to `fileChange` would be the
  invented taxonomy this SDK refuses everywhere else. Nor are PATHS: a call
  that wrote /Users/x/proj/recipe.txt says so, and that is where it was
  written. Across machines the receiver must be told where things are now
  (or the workspace copied to the same path) — as any colleague would be.
- reasoning: a user-role handoff note after the turn it belonged to — where
  the destination accepts it. Measured: fed back as the agent's OWN thoughts
  it is rejected (Claude signs thinking blocks) or silently ignored (codex
  accepts a reasoning item and never reads it). As a note, codex uses it.
  Claude does NOT accept it in any form: Anthropic's safeguards flag real
  model reasoning replayed as user content (`[reasoning_extraction]`) —
  hand-written notes passed, genuine Opus thinking was refused. So the
  Claude renderer omits reasoning, and the worker warns with the count,
  because dropping context silently is the one thing this SDK never does.
  Pi omits it too, for both reasons: pi drops thinking another model wrote,
  and pi often runs Claude.
- files: images go to every harness, as the image input each takes (pi
  takes bytes, so an image known only by URL becomes a note naming it);
  PDFs to Claude as documents. Anything else (a PDF into codex, a text
  file) is left out and counted (`unsupported_files`), and the worker
  warns, like it does for reasoning into Claude.
"""

from __future__ import annotations

import base64
import json
import warnings
from collections.abc import Sequence
from typing import Any

from ..models.core.helpers import files
from ..types import events, media
from ..types import messages as messages_
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


def check_parts(
    message: messages_.Message, kind: str, accepts: Sequence[str]
) -> None:
    """Refuse a part the CLI cannot take, before anything runs.

    Text parts go everywhere. Text files (`text/*`) go everywhere, as text.
    `accepts` lists the other media type prefixes the CLI takes.
    """
    for part in message.parts:
        if isinstance(part, messages_.TextPart):
            continue
        if not isinstance(part, messages_.FilePart):
            raise errors.UnsupportedError(
                f"{kind} does not take {part.kind} parts in a user message"
            )
        if not any(part.media_type.startswith(p) for p in ("text/", *accepts)):
            raise errors.UnsupportedError(
                f"{kind} does not take {part.media_type} files"
            )


# -- the CLI's own tools as events --------------------------------------------


def builtin_call(
    tool_call_id: str,
    tool_name: str,
    tool_args: str,
    *,
    started: set[str],
) -> list[
    events.BuiltinToolStart | events.BuiltinToolDelta | events.BuiltinToolEnd
]:
    """Return the events of one finished CLI tool call.

    `started` holds the calls whose start was already sent: the start is
    sent here otherwise, because the hydrator only completes a call it saw
    start. The arguments travel as a delta: the hydrator builds a call's
    arguments only from its deltas.
    """
    out: list[
        events.BuiltinToolStart
        | events.BuiltinToolDelta
        | events.BuiltinToolEnd
    ] = []
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


# -- seeding a CLI with a past -------------------------------------------------


def result_text(value: Any) -> str:
    """Return a tool result as the text a CLI's store holds."""
    return value if isinstance(value, str) else json.dumps(value)


def warn_unsupported_files(
    messages: list[messages_.Message],
    kind: str,
    *,
    accepts: Sequence[str],
    takes: str,
) -> None:
    """Say how many files a history held that `kind` was not given.

    `accepts` lists the media type prefixes `kind` takes from a history, and
    `takes` names them for the warning.
    """
    left = sum(
        1
        for m in messages
        for p in m.parts
        if isinstance(p, messages_.FilePart)
        and not any(p.media_type.startswith(a) for a in accepts)
    )
    if left:
        warnings.warn(
            f"{left} file(s) in the history were not carried into {kind}: "
            f"it takes {takes} from a history. Text, tool calls and images "
            "were carried.",
            stacklevel=3,
        )


def reasoning_count(messages: list[messages_.Message]) -> int:
    return sum(
        1
        for m in messages
        for p in m.parts
        if isinstance(p, messages_.ReasoningPart) and p.text.strip()
    )
