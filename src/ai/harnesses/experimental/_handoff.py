"""Starting a conversation FROM a history: one rendering rule, both harnesses.

A conversation is a `list[Message]` — what `history()` returns and what
`Result.messages` carries. To move one, read it as messages and start a new
session from them. Each harness stores it natively so the agent truly has
the context, and the rule for what becomes what is the same everywhere:

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
  Claude renderer omits reasoning, and the adapter warns with the count,
  because dropping context silently is the one thing this SDK never does.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any

from ...types import messages as messages_

NOTE_PREFIX = "[Handoff from a previous session — earlier reasoning: "


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _note(reasoning: list[str]) -> str:
    return (
        NOTE_PREFIX + " ".join(r.strip() for r in reasoning if r.strip()) + "]"
    )


def to_codex_items(messages: list[messages_.Message]) -> list[dict[str, Any]]:
    """Responses API items for `thread/inject_items`."""
    items: list[dict[str, Any]] = []
    for m in messages:
        reasoning: list[str] = []
        for p in m.parts:
            if isinstance(p, messages_.TextPart) and p.text.strip():
                ctype = "input_text" if m.role == "user" else "output_text"
                items.append(
                    {
                        "type": "message",
                        "role": m.role,
                        "content": [{"type": ctype, "text": p.text}],
                    }
                )
            elif isinstance(p, messages_.ToolCallPart):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": p.tool_call_id or p.id,
                        "name": p.tool_name,
                        "arguments": p.tool_args or "{}",
                    }
                )
            elif isinstance(p, messages_.ToolResultPart):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": p.tool_call_id,
                        "output": _text(p.result),
                    }
                )
            elif isinstance(p, messages_.ReasoningPart) and p.text.strip():
                reasoning.append(p.text)
        if reasoning:
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _note(reasoning)}
                    ],
                }
            )
    return items


def reasoning_count(messages: list[messages_.Message]) -> int:
    return sum(
        1
        for m in messages
        for p in m.parts
        if isinstance(p, messages_.ReasoningPart) and p.text.strip()
    )


def to_claude_records(
    messages: list[messages_.Message], session_id: str, cwd: str
) -> list[dict[str, Any]]:
    """Build Claude Code transcript records for the CLI's store.

    Chained by parentUuid.

    Reasoning parts are omitted: see the module docstring. Callers report
    the omission — `reasoning_count()` says how many.
    """
    records: list[dict[str, Any]] = []
    parent: str | None = None

    def add(role: str, content: Any) -> None:
        nonlocal parent
        uid = str(uuid.uuid4())
        ts = (
            datetime.datetime.now(datetime.UTC)
            .isoformat()
            .replace("+00:00", "Z")
        )
        records.append(
            {
                "parentUuid": parent,
                "isSidechain": False,
                "type": role,
                "message": {"role": role, "content": content},
                "uuid": uid,
                "timestamp": ts,
                "sessionId": session_id,
                "cwd": cwd,
            }
        )
        parent = uid

    for m in messages:
        blocks: list[dict[str, Any]] = []
        for p in m.parts:
            if isinstance(p, messages_.TextPart) and p.text.strip():
                blocks.append({"type": "text", "text": p.text})
            elif isinstance(p, messages_.ToolCallPart):
                try:
                    args = json.loads(p.tool_args) if p.tool_args else {}
                except json.JSONDecodeError:
                    args = {"raw": p.tool_args}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": p.tool_call_id or p.id,
                        "name": p.tool_name,
                        "input": args,
                    }
                )
            elif isinstance(p, messages_.ToolResultPart):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": p.tool_call_id,
                        "content": _text(p.result),
                        **(
                            {"is_error": True}
                            if p.result_kind == "error"
                            else {}
                        ),
                    }
                )
            # ReasoningPart: omitted on purpose — Anthropic's safeguards
            # refuse replayed reasoning. The adapter warns with the count.
        if blocks:
            # Tool results live in role="tool" messages in the IR; on disk
            # the CLI keeps them as user records with tool_result blocks.
            add("user" if m.role in ("user", "tool") else "assistant", blocks)
    return records
