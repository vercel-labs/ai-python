"""Seeding pi with a past: `AgentMessage`s for a session file.

The rule for what becomes what is the same for every CLI: see `_handoff`.
"""

from __future__ import annotations

import json
from typing import Any

from ...types import media
from ...types import messages as messages_
from .. import _handoff

# What names a reply's author in a pi session: its API and model.
AUTHOR = ("api", "provider", "model")
SYSTEM_PREFIX = "[Handoff from a previous session — instructions it was given: "


def to_pi_messages(
    messages: list[messages_.Message], *, timestamp: int
) -> list[dict[str, Any]]:
    """Build pi `AgentMessage`s for a session file.

    Assistant messages pi wrote keep their author (`provider_metadata["pi"]`):
    pi restores a session's model from it. Any other is attributed to a
    model pi does not run (`ai-python/handoff`), so pi treats it as another
    model's — it keeps its text and tool calls and normalizes the call ids
    for its own provider. Consecutive assistant messages are merged, the way one
    model response holds text and calls together. Reasoning is omitted: see
    the module docstring.
    """
    out: list[dict[str, Any]] = []
    usage = {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "total": 0,
        },
    }
    for m in _handoff.harness_view(messages):
        if m.role == "tool":
            for p in m.parts:
                if isinstance(p, messages_.ToolResultPart):
                    out.append(
                        {
                            "role": "toolResult",
                            "toolCallId": p.tool_call_id,
                            "toolName": p.tool_name,
                            "content": [
                                {
                                    "type": "text",
                                    "text": _handoff.result_text(p.result),
                                }
                            ],
                            "isError": p.result_kind == "error",
                            "timestamp": timestamp,
                        }
                    )
            continue
        blocks: list[dict[str, Any]] = []
        for p in m.parts:
            if isinstance(p, messages_.TextPart) and p.text.strip():
                blocks.append({"type": "text", "text": p.text})
            elif (
                isinstance(p, messages_.FilePart)
                and m.role == "user"
                and p.media_type.startswith("image/")
            ):
                if isinstance(p.data, str) and media.is_downloadable_url(
                    p.data
                ):
                    blocks.append(
                        {"type": "text", "text": f"[image: {p.data}]"}
                    )
                else:
                    blocks.append(
                        {
                            "type": "image",
                            "data": media.data_to_base64(p.data),
                            "mimeType": p.media_type,
                        }
                    )
            elif (
                isinstance(p, messages_.ToolCallPart) and m.role == "assistant"
            ):
                try:
                    args = json.loads(p.tool_args) if p.tool_args else {}
                except json.JSONDecodeError:
                    args = {"raw": p.tool_args}
                blocks.append(
                    {
                        "type": "toolCall",
                        "id": p.tool_call_id or p.id,
                        "name": p.tool_name,
                        "arguments": args if isinstance(args, dict) else {},
                    }
                )
        if not blocks:
            continue
        if m.role == "system":
            # Measured (pi 1.3.0): a system message in a session file is
            # pi's own prompt state, and replaying one REPLACED pi's system
            # prompt and its tools. So instructions travel as what they are
            # to the agent — the caller's words — never as its own output.
            text = " ".join(b["text"] for b in blocks if b["type"] == "text")
            blocks = [{"type": "text", "text": SYSTEM_PREFIX + text + "]"}]
        if m.role in ("user", "system"):
            out.append(
                {"role": "user", "content": blocks, "timestamp": timestamp}
            )
            continue
        if out and out[-1]["role"] == "assistant":
            out[-1]["content"] += blocks
        else:
            # A reply pi wrote keeps its author: pi restores a session's model
            # from its last reply's, and a model it does not know (ours)
            # put the conversation on pi's default one (measured).
            author = (m.provider_metadata or {}).get("pi") or {}
            known = all(isinstance(author.get(k), str) for k in AUTHOR)
            out.append(
                {
                    "role": "assistant",
                    "content": blocks,
                    **(
                        {k: author[k] for k in AUTHOR}
                        if known
                        else {
                            "api": "ai-python",
                            "provider": "ai-python",
                            "model": "handoff",
                        }
                    ),
                    "usage": usage,
                    "stopReason": "stop",
                    "timestamp": timestamp,
                }
            )
        if any(b["type"] == "toolCall" for b in out[-1]["content"]):
            out[-1]["stopReason"] = "toolUse"
    return out
