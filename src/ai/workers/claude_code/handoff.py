"""Seeding Claude Code with a past: records for its transcript store.

The rule for what becomes what is the same for every CLI: see `_convert`.
"""

from __future__ import annotations

import base64
import datetime
import json
import uuid
from typing import Any

from ...types import messages as messages_
from .. import _convert


async def to_claude_records(
    messages: list[messages_.Message], session_id: str, cwd: str
) -> list[dict[str, Any]]:
    """Build Claude Code transcript records for the CLI's store.

    Chained by parentUuid. Reasoning parts are omitted: see `_convert`.
    Callers report the omission; `_convert.reasoning_count()` says how many.
    Files go in as base64, a URL downloaded first: Claude Code drops an
    image given by URL without a word (measured on prompts).
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
        role = "user" if m.role in ("user", "tool") else "assistant"
        blocks: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for p in m.parts:
            if isinstance(p, messages_.BuiltinToolReturnPart):
                # A tool the CLI ran: the worker holds its result in the
                # assistant message, the store in the NEXT user record. So
                # the assistant record ends here, and the results follow.
                if blocks:
                    add(role, blocks)
                    blocks = []
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": p.tool_call_id,
                        "content": _convert.result_text(p.result),
                        **({"is_error": True} if p.is_error else {}),
                    }
                )
                continue
            if results:
                add("user", results)
                results = []
            if isinstance(p, messages_.TextPart) and p.text.strip():
                blocks.append({"type": "text", "text": p.text})
            elif isinstance(p, messages_.FilePart) and m.role == "user":
                if p.media_type.startswith("image/"):
                    kind = "image"
                elif p.media_type == "application/pdf":
                    kind = "document"
                else:
                    continue  # counted by `_convert.warn_unsupported_files`
                encoded = base64.b64encode(await _convert.data(p)).decode()
                blocks.append(
                    {
                        "type": kind,
                        "source": {
                            "type": "base64",
                            "media_type": p.media_type,
                            "data": encoded,
                        },
                    }
                )
            elif isinstance(
                p, messages_.ToolCallPart | messages_.BuiltinToolCallPart
            ):
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
                # A tool an `ai.Agent` ran: its result is in a `tool`
                # message, which becomes a user record of its own.
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": p.tool_call_id,
                        "content": _convert.result_text(p.result),
                        **(
                            {"is_error": True}
                            if p.result_kind == "error"
                            else {}
                        ),
                    }
                )
            # ReasoningPart: omitted on purpose. Anthropic's safeguards
            # refuse replayed reasoning. The wire warns with the count.
        if blocks:
            add(role, blocks)
        if results:
            add("user", results)
    return records
