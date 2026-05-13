"""Session history — persist and resume conversations.

Sessions are stored as JSONL files under ``.tau/sessions/``.  Each line
is a JSON-serialised ``ai.messages.Message``.  The first line is always
a metadata object (not a Message) carrying session-level info:

    {"meta": true, "session_id": "...", "model": "...", "cwd": "...", "created": "..."}

Usage:
    # New session (default)
    uv run python tau.py

    # Resume the most recent session
    uv run python tau.py --resume

    # Resume a specific session by ID (or prefix)
    uv run python tau.py --session 20250101-120000

    # List saved sessions
    uv run python tau.py --list
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import UTC, datetime
from typing import Any

import ai

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SESSIONS_DIR = pathlib.Path(".tau") / "sessions"


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Session ID
# ---------------------------------------------------------------------------


def new_session_id() -> str:
    """Timestamp-based, human-readable session ID."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _meta_line(session_id: str, model: str) -> str:
    return json.dumps(
        {
            "meta": True,
            "session_id": session_id,
            "model": model,
            "cwd": os.getcwd(),
            "created": datetime.now(UTC).isoformat(),
        },
        ensure_ascii=False,
    )


def _read_meta(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
            if first:
                obj = json.loads(first)
                if isinstance(obj, dict) and obj.get("meta"):
                    return obj
    except (OSError, json.JSONDecodeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def create_session(session_id: str, model: str) -> pathlib.Path:
    """Create a new JSONL session file and write the metadata header."""
    _ensure_dir()
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(_meta_line(session_id, model) + "\n")
    return path


def append_messages(
    path: pathlib.Path,
    messages: list[ai.messages.Message],
    *,
    after: int = 0,
) -> int:
    """Append new messages to the session file.

    ``after`` is the count of messages already written (excluding the
    metadata line).  Only messages from ``messages[after:]`` are
    appended.  Returns the new total written count.
    """
    new = messages[after:]
    if not new:
        return after
    with path.open("a", encoding="utf-8") as f:
        for msg in new:
            f.write(msg.model_dump_json() + "\n")
    return after + len(new)


# ---------------------------------------------------------------------------
# Reading / resuming
# ---------------------------------------------------------------------------


def list_sessions() -> list[dict[str, Any]]:
    """Return metadata dicts for all sessions, newest first."""
    _ensure_dir()
    sessions: list[dict[str, Any]] = []
    for p in sorted(SESSIONS_DIR.glob("*.jsonl"), reverse=True):
        meta = _read_meta(p)
        if meta is not None:
            meta["_path"] = str(p)
            sessions.append(meta)
    return sessions


def resolve_session(session_id: str | None) -> pathlib.Path | None:
    """Find a session file.

    - ``None`` → most recent session
    - exact match → that session
    - prefix match → first match (newest first)
    """
    _ensure_dir()
    files = sorted(SESSIONS_DIR.glob("*.jsonl"), reverse=True)
    if not files:
        return None
    if session_id is None:
        return files[0]
    # Exact
    exact = SESSIONS_DIR / f"{session_id}.jsonl"
    if exact.exists():
        return exact
    # Prefix
    for f in files:
        if f.stem.startswith(session_id):
            return f
    return None


def load_messages(
    path: pathlib.Path,
) -> tuple[dict[str, Any], list[ai.messages.Message]]:
    """Load session metadata + messages from a JSONL file.

    Returns ``(meta_dict, messages_list)``.  The system message is
    included in the list (it's persisted like any other message).
    """
    meta: dict[str, Any] = {}
    messages: list[ai.messages.Message] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if lineno == 1:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("meta"):
                    meta = obj
                    continue
            messages.append(ai.messages.Message.model_validate_json(line))
    return meta, messages
