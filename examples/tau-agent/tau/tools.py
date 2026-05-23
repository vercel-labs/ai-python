"""tau's coding tools — pi's seven built-ins, plain Python.

Mirrors pi's tool surface (read, write, edit, bash, grep, find, ls) so
the model gets the same affordances.

All tools use ``require_approval=True`` so the ``ai`` library fires a
hook for every invocation.  The ``ApprovalTracker`` in ``tau.app``
auto-approves safe cases (e.g. reads under cwd) without prompting the
operator; this is the only way to get hooks — the library doesn't
support per-invocation gating otherwise.

No workspace jail — paths resolve against the process cwd and the
host (or the approval flow) is what keeps things in line.
"""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import re
from collections.abc import AsyncGenerator
from typing import Annotated, Any, Literal

import ai
import ai.agents
import ai.types.events
import pydantic

# Image formats we support (subset that models typically accept).
_IMAGE_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)


def _detect_image_mime(path: pathlib.Path) -> str | None:
    """Read magic bytes and return a supported image MIME type, or None."""
    try:
        header = path.read_bytes()[:32]
    except OSError:
        return None
    mime = ai.types.media.detect_image_media_type(header)
    return mime if mime in _IMAGE_MIME_TYPES else None


# ---------------------------------------------------------------------------
# Truncation — match pi's defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50 KB
GREP_MAX_LINE_LENGTH = 500

# Directories grep/find skip by default.  No .gitignore support — this
# is the cheap approximation.
EXCLUDE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".next",
        ".turbo",
    }
)


@dataclasses.dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: Literal["lines", "bytes"] | None
    total_lines: int
    output_lines: int
    total_bytes: int
    output_bytes: int
    first_line_exceeds_limit: bool = False


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep complete lines from the start until a cap is hit."""
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            output_lines=total_lines,
            total_bytes=total_bytes,
            output_bytes=total_bytes,
        )

    if len(lines[0].encode("utf-8")) > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            output_lines=0,
            total_bytes=total_bytes,
            output_bytes=0,
            first_line_exceeds_limit=True,
        )

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        out_lines.append(line)
        out_bytes += line_bytes

    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"

    out = "\n".join(out_lines)
    return TruncationResult(
        content=out,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        output_lines=len(out_lines),
        total_bytes=total_bytes,
        output_bytes=len(out.encode("utf-8")),
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep complete lines from the end until a cap is hit.

    Used for bash output — errors and final results sit at the bottom.
    """
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            output_lines=total_lines,
            total_bytes=total_bytes,
            output_bytes=total_bytes,
        )

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by: Literal["lines", "bytes"] = "lines"
    for line in reversed(lines):
        if len(out_lines) >= max_lines:
            break
        line_bytes = len(line.encode("utf-8")) + (1 if out_lines else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        out_lines.append(line)
        out_bytes += line_bytes

    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"

    out_lines.reverse()
    out = "\n".join(out_lines)
    return TruncationResult(
        content=out,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        output_lines=len(out_lines),
        total_bytes=total_bytes,
        output_bytes=len(out.encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


@ai.tool(require_approval=True)
async def read(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> Any:
    """Read the contents of a file.

    Output is truncated to 2000 lines or 50KB (whichever is hit first).
    Use offset/limit for large files; when truncated, the result ends
    with a "Use offset=N to continue" hint.  offset is 1-indexed.

    Supports image files (jpg, png, gif, webp) which are returned as
    base64-encoded data via FilePart objects.
    """
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"Not a file: {path}")

    # Image files: return a text label + FilePart so providers emit
    # a real image content block instead of stringified JSON.
    mime = _detect_image_mime(p)
    if mime is not None:
        data = p.read_bytes()
        size_str = format_size(len(data))
        return [
            f"Read image file [{mime}, {size_str}]",
            ai.file_part(data, media_type=mime),
        ]

    text = p.read_text(encoding="utf-8", errors="replace")
    all_lines = text.split("\n")
    total_lines = len(all_lines)

    start = (offset - 1) if offset else 0  # 0-indexed
    if start >= total_lines:
        raise ValueError(
            f"Offset {offset} is beyond end of file ({total_lines} lines total)"
        )
    start_display = start + 1

    if limit is not None:
        end = min(start + limit, total_lines)
        selected = "\n".join(all_lines[start:end])
        user_limited = True
        user_end_display = end  # 1-indexed inclusive
    else:
        selected = "\n".join(all_lines[start:])
        user_limited = False
        user_end_display = total_lines

    tr = truncate_head(selected)

    if tr.first_line_exceeds_limit:
        first_size = format_size(len(all_lines[start].encode("utf-8")))
        return (
            f"[Line {start_display} is {first_size}, exceeds "
            f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: "
            f"sed -n '{start_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
        )

    out = tr.content
    if tr.truncated:
        end_display = start + tr.output_lines  # 1-indexed inclusive
        next_offset = end_display + 1
        if tr.truncated_by == "lines":
            out += (
                f"\n\n[Showing lines {start_display}-{end_display} of "
                f"{total_lines}. Use offset={next_offset} to continue.]"
            )
        else:
            out += (
                f"\n\n[Showing lines {start_display}-{end_display} of "
                f"{total_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                f"Use offset={next_offset} to continue.]"
            )
    elif user_limited and user_end_display < total_lines:
        remaining = total_lines - user_end_display
        next_offset = user_end_display + 1
        out += (
            f"\n\n[{remaining} more lines in file. "
            f"Use offset={next_offset} to continue.]"
        )

    return out


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


@ai.tool(require_approval=True)
async def write(path: str, content: str) -> str:
    """Write content to a file.

    Creates the file if it doesn't exist, overwrites if it does.
    Automatically creates parent directories.  Use write only for new
    files or complete rewrites — use edit for targeted changes.
    """
    p = pathlib.Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TextEdit(pydantic.BaseModel):
    """A single targeted str_replace edit."""

    oldText: str = pydantic.Field(
        description=(
            "Exact text for one targeted replacement. It must be unique "
            "in the original file and must not overlap with any other "
            "edits[].oldText in the same call."
        )
    )
    newText: str = pydantic.Field(
        description="Replacement text for this targeted edit."
    )


def edit_string(
    content: str,
    edits: list[TextEdit],
    filename: str = "<string>",
) -> str:
    """Apply edits to a string and return the result.

    Each edit's ``oldText`` must match exactly once in *content*.
    Edits are resolved against the original content and must not overlap.
    """
    spans: list[tuple[int, int, str, int]] = []  # start, end, new, idx
    for i, e in enumerate(edits):
        if not e.oldText:
            raise ValueError(f"edits[{i}].oldText is empty")
        count = content.count(e.oldText)
        if count == 0:
            raise ValueError(f"edits[{i}].oldText not found in {filename}")
        if count > 1:
            raise ValueError(
                f"edits[{i}].oldText matches {count} times "
                f"in {filename}; must be unique"
            )
        pos = content.index(e.oldText)
        spans.append((pos, pos + len(e.oldText), e.newText, i))

    spans.sort()
    for j in range(1, len(spans)):
        if spans[j][0] < spans[j - 1][1]:
            raise ValueError(
                f"edits[{spans[j - 1][3]}] and edits[{spans[j][3]}] overlap"
            )

    result = content
    for start, end, new_text, _ in reversed(spans):
        result = result[:start] + new_text + result[end:]
    return result


class EditResult(pydantic.BaseModel):
    """Structured result from the edit tool.

    Persisted in the session so the UI can render a diff on restore.
    The model only sees ``message`` via the aggregator's
    ``to_model_input``.
    """

    message: str
    old_content: str
    new_content: str


class _EditAggregator(
    ai.types.events.Aggregator[EditResult, EditResult, str],
):
    def __init__(self) -> None:
        self._result: EditResult | None = None

    def feed(self, item: EditResult) -> None:
        self._result = item

    def snapshot(self) -> EditResult:
        assert self._result is not None
        return self._result

    @classmethod
    def to_model_input(cls, snapshot: EditResult) -> str:
        if isinstance(snapshot, dict):
            return snapshot.get("message", str(snapshot))
        return snapshot.message


type _EditTool = Annotated[
    AsyncGenerator[EditResult],
    ai.agents.Aggregate(_EditAggregator),
]


@ai.tool(require_approval=True)
async def edit(path: str, edits: list[TextEdit]) -> _EditTool:
    """Edit a single file using exact text replacement.

    Every edits[].oldText must match a unique, non-overlapping region of
    the original file.  Each oldText is matched against the ORIGINAL
    file, not after earlier edits are applied; emit one call with
    multiple disjoint edits rather than several calls.
    """
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"Not a file: {path}")
    if not edits:
        raise ValueError("edits must be non-empty")

    old_content = p.read_text(encoding="utf-8")
    new_content = edit_string(old_content, edits, path)
    p.write_text(new_content, encoding="utf-8")
    yield EditResult(
        message=f"Successfully replaced {len(edits)} block(s) in {path}.",
        old_content=old_content,
        new_content=new_content,
    )


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


@ai.tool(require_approval=True)
async def bash(
    command: str, timeout: float | None = None
) -> ai.StreamingTextTool:
    """Execute a bash command in the current working directory.

    Returns stdout and stderr.  Output is truncated to the last 2000
    lines or 50KB (whichever is hit first).  Optionally provide a
    timeout in seconds.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    buf: list[str] = []
    timed_out = False
    deadline = (
        asyncio.get_event_loop().time() + timeout
        if timeout is not None
        else None
    )
    try:
        async for raw in proc.stdout:
            if (
                deadline is not None
                and asyncio.get_event_loop().time() > deadline
            ):
                raise TimeoutError
            line = raw.decode("utf-8", errors="replace")
            buf.append(line)
            yield line
    except TimeoutError:
        timed_out = True
        proc.kill()
        await proc.wait()
        yield f"\n[Timed out after {timeout}s]\n"
        return

    await proc.wait()

    # After streaming, check if we need to append metadata.
    text = "".join(buf)
    tr = truncate_tail(text)
    if tr.truncated:
        if tr.truncated_by == "lines":
            yield (
                f"\n\n[Truncated: showing last {tr.output_lines} of "
                f"{tr.total_lines} lines]"
            )
        else:
            yield (
                f"\n\n[Truncated: showing last {format_size(tr.output_bytes)} "
                f"of {format_size(tr.total_bytes)}]"
            )

    if not timed_out and proc.returncode and proc.returncode != 0:
        yield f"\n\n[Exit code: {proc.returncode}]"

    if not buf and not timed_out:
        yield "[no output]"


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


@ai.tool(require_approval=True)
async def grep(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    ignore_case: bool = False,
    literal: bool = False,
    context: int = 0,
    limit: int = 100,
) -> str:
    """Search file contents for a pattern.

    Returns matching lines as ``path:lineno:content``.  Skips common
    cruft directories (.git, node_modules, __pycache__, etc.) but does
    NOT respect .gitignore.  Output is truncated to ``limit`` matches
    or 50KB.  Long match lines are truncated to 500 chars.
    """
    base = pathlib.Path(path).expanduser() if path else pathlib.Path.cwd()
    if not base.exists():
        raise FileNotFoundError(f"No such path: {path}")

    flags = re.IGNORECASE if ignore_case else 0
    raw = re.escape(pattern) if literal else pattern
    try:
        pat = re.compile(raw, flags)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}") from e

    if base.is_file():
        files: list[pathlib.Path] = [base]
    else:
        candidates = base.rglob(glob) if glob else base.rglob("*")
        files = []
        for f in candidates:
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(base)
            except ValueError:
                continue
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            files.append(f)

    hits: list[str] = []
    bytes_used = 0
    stopped_by: Literal["limit", "bytes", None] = None

    def _short(line: str) -> str:
        if len(line) <= GREP_MAX_LINE_LENGTH:
            return line
        return line[:GREP_MAX_LINE_LENGTH] + "... [truncated]"

    for f in sorted(files):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel_path = f.relative_to(base)
        except ValueError:
            rel_path = f
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if not pat.search(line):
                continue
            if context > 0:
                ctx_chunks = []
                for j in range(
                    max(0, i - context), min(len(lines), i + context + 1)
                ):
                    sep = ":" if j == i else "-"
                    ctx_chunks.append(
                        f"{rel_path}{sep}{j + 1}{sep}{_short(lines[j])}"
                    )
                entry = "\n".join(ctx_chunks)
            else:
                entry = f"{rel_path}:{i + 1}:{_short(line)}"
            hits.append(entry)
            bytes_used += len(entry.encode("utf-8")) + 1
            if len(hits) >= limit:
                stopped_by = "limit"
                break
            if bytes_used > DEFAULT_MAX_BYTES:
                stopped_by = "bytes"
                break
        if stopped_by:
            break

    if not hits:
        return "No matches found."

    out = "\n".join(hits)
    if stopped_by == "limit":
        out += f"\n\n[Stopped at {len(hits)} matches; raise limit to see more]"
    elif stopped_by == "bytes":
        out += f"\n\n[Stopped at {format_size(DEFAULT_MAX_BYTES)} of output]"
    return out


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


@ai.tool(require_approval=True)
async def find(
    pattern: str,
    path: str | None = None,
    limit: int = 1000,
) -> str:
    """Search for files by glob pattern.

    Returns matching file paths relative to the search directory.
    Skips common cruft directories but does NOT respect .gitignore.
    Output is truncated to ``limit`` results or 50KB.
    """
    base = pathlib.Path(path).expanduser() if path else pathlib.Path.cwd()
    if not base.exists():
        raise FileNotFoundError(f"No such path: {path}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    matches: list[str] = []
    bytes_used = 0
    stopped_by: Literal["limit", "bytes", None] = None
    for p in base.rglob(pattern):
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        s = str(rel) + ("/" if p.is_dir() else "")
        matches.append(s)
        bytes_used += len(s.encode("utf-8")) + 1
        if len(matches) >= limit:
            stopped_by = "limit"
            break
        if bytes_used > DEFAULT_MAX_BYTES:
            stopped_by = "bytes"
            break

    if not matches:
        return "No matches found."

    out = "\n".join(sorted(matches))
    if stopped_by == "limit":
        out += f"\n\n[Stopped at {limit} matches; raise limit to see more]"
    elif stopped_by == "bytes":
        out += f"\n\n[Stopped at {format_size(DEFAULT_MAX_BYTES)} of output]"
    return out


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


@ai.tool(require_approval=True)
async def ls(path: str | None = None, limit: int = 500) -> str:
    """List directory contents.

    Entries are sorted alphabetically; directories are suffixed with
    ``/``.  Includes dotfiles.  Output is truncated to ``limit``
    entries.
    """
    p = pathlib.Path(path).expanduser() if path else pathlib.Path.cwd()
    if not p.exists():
        raise FileNotFoundError(f"No such path: {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries: list[str] = []
    for entry in sorted(p.iterdir(), key=lambda e: e.name):
        name = entry.name + ("/" if entry.is_dir() else "")
        entries.append(name)
        if len(entries) >= limit:
            break

    if not entries:
        return "(empty)"

    out = "\n".join(entries)
    if len(entries) >= limit:
        out += f"\n\n[Stopped at {limit} entries; raise limit to see more]"
    return out


# ---------------------------------------------------------------------------
# Tool set
# ---------------------------------------------------------------------------

TOOLS = [read, write, edit, bash, grep, find, ls]

__all__ = [
    "TOOLS",
    "bash",
    "edit",
    "find",
    "grep",
    "ls",
    "read",
    "write",
]
