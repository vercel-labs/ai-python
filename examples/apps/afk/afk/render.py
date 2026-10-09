"""A transcript, drawn for a terminal: what `afk peek` shows.

Readable at a glance, the way the harness's own TUI reads: your messages
marked with a chevron, the agent's words wrapped to the terminal, each tool
call on one line with the argument that says what it does, and its result as one
line under it. Long text is cut, not dropped silently: a cut says so.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

MAX_USER_LINES = 3
MAX_TEXT_LINES = 12
TOOL_ARG_KEYS = (
    "command",
    "cmd",
    "file_path",
    "path",
    "pattern",
    "url",
    "query",
    "description",
    "prompt",
)
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


class Style:
    def __init__(self, *, color: bool) -> None:
        on = color
        self.dim = (lambda s: f"\x1b[2m{s}\x1b[0m") if on else (lambda s: s)
        self.bold = (lambda s: f"\x1b[1m{s}\x1b[0m") if on else (lambda s: s)
        self.red = (lambda s: f"\x1b[31m{s}\x1b[0m") if on else (lambda s: s)


def lines(message: Any, width: int, style: Style) -> list[str]:
    """The terminal lines for one transcript message; [] for nothing to show."""
    out: list[str] = []
    for part in message.parts:
        kind = getattr(part, "kind", "")
        if kind == "text":
            text = _REMINDER.sub("", part.text).replace("**", "").strip()
            if not text:
                continue
            if message.role == "user":
                out += [
                    style.bold(ln)
                    for ln in _wrap(
                        text,
                        width,
                        "› ",  # noqa: RUF001
                        "  ",
                        MAX_USER_LINES,
                    )
                ]
            else:
                out += _wrap(text, width, "  ", "  ", MAX_TEXT_LINES)
        elif kind == "tool_call":
            out.append(
                style.dim(
                    _clip(
                        (
                            f"  ⏺ {part.tool_name} "
                            f"{_tool_arg(part.tool_args)}"
                        ).rstrip(),
                        width,
                    )
                )
            )
        elif kind == "tool_result":
            summary, failed = _result(part)
            line = _clip(f"    ⎿ {summary}", width)
            out.append(style.red(line) if failed else style.dim(line))
    return out


def _wrap(
    text: str, width: int, first: str, rest: str, limit: int
) -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            if wrapped and wrapped[-1] != "":
                wrapped.append("")
            continue
        wrapped += textwrap.wrap(paragraph, max(20, width - len(first))) or [""]
    while wrapped and wrapped[-1] == "":
        wrapped.pop()
    cut = len(wrapped) - limit
    shown = wrapped[:limit]
    result = [(first if i == 0 else rest) + ln for i, ln in enumerate(shown)]
    if cut > 0:
        result.append(f"{rest}… {cut} more line{'s' * (cut != 1)}")
    return result


def _tool_arg(raw: str) -> str:
    """The one argument that says what a call does: its command, its file…"""
    try:
        args = json.loads(raw) if raw else {}
    except ValueError:
        return _one_line(raw)
    if isinstance(args, dict):
        for key in TOOL_ARG_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return _one_line(value)
            if (
                isinstance(value, list)
                and value
                and all(isinstance(v, str) for v in value)
            ):
                return _one_line(" ".join(value))
        for value in args.values():
            if isinstance(value, str) and value.strip():
                return _one_line(value)
    return ""


def _result(part: Any) -> tuple[str, bool]:
    failed = getattr(part, "result_kind", "") == "error"
    text = _text_of(part.result).strip()
    if not text:
        return ("error" if failed else "done"), failed
    all_lines = [ln for ln in text.splitlines() if ln.strip()]
    first = _one_line(all_lines[0]) if all_lines else ""
    more = len(all_lines) - 1
    summary = first + (
        f"  (+{more} line{'s' * (more != 1)})" if more > 0 else ""
    )
    return ("error: " + summary if failed else summary), failed


def _text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text_of(v) for v in value)
    if isinstance(value, dict):
        for key in ("text", "output", "content", "stdout", "result"):
            if key in value:
                return _text_of(value[key])
        return json.dumps(value)
    return str(value)


def _one_line(s: str) -> str:
    return " ".join(s.split())


def _clip(s: str, width: int) -> str:
    return s if len(s) <= width else s[: max(1, width - 1)] + "…"
