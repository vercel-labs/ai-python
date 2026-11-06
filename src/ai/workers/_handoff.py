"""Seeding a CLI with a past: one rendering rule, every CLI.

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
  Claude renderer omits reasoning, and the adapter warns with the count,
  because dropping context silently is the one thing this SDK never does.
  Pi omits it too, for both reasons: pi drops thinking another model wrote,
  and pi often runs Claude.
- files: images go to every harness, as the image input each takes (pi
  takes bytes, so an image known only by URL becomes a note naming it);
  PDFs to Claude as documents. Anything else (a PDF into codex, a text
  file) is left out and counted (`unsupported_files`), and the adapter
  warns, like it does for reasoning into Claude.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

from ..types import messages as messages_


def result_text(value: Any) -> str:
    """Return a tool result as the text a CLI's store holds."""
    return value if isinstance(value, str) else json.dumps(value)


# -- the two views of a past ---------------------------------------------------
#
# A worker's messages hold two kinds of tool calls. The tools an `ai.Agent`
# runs (through the bridge) are `ToolCallPart`s with their results in `tool`
# messages, like a model's. The tools the CLI runs itself are built-in
# parts, with the result in the same assistant message, like a provider's
# web search. A CLI's store holds both kinds the same way: a call, then its
# result in a message of its own. `harness_view` gives the renderers that
# one shape, so each renderer handles one kind of tool call, not two;
# `model_view` is its reverse, for a history read back from a store.


def harness_view(history: list[messages_.Message]) -> list[messages_.Message]:
    """Return a past with the CLI's own tools as ordinary tool calls.

    A worker's messages hold the tools a CLI ran itself as
    `BuiltinToolCallPart`/`BuiltinToolReturnPart`, the way a provider's web
    search is. A CLI's store holds them as tool calls, each with its result
    in a `tool` message after it. The renderers below read that shape.
    """
    out: list[messages_.Message] = []
    for message in history:
        parts: list[messages_.Part] = []
        results: list[messages_.Part] = []
        first = True
        for part in message.parts:
            if results and not isinstance(
                part, messages_.BuiltinToolReturnPart
            ):
                # What follows a result comes after it: a new message.
                if parts:
                    out.append(_split(message, parts, first=first))
                    first = False
                out.append(messages_.Message(role="tool", parts=results))
                parts, results = [], []
            if isinstance(part, messages_.BuiltinToolCallPart):
                parts.append(
                    messages_.ToolCallPart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        tool_args=part.tool_args or "{}",
                        provider_metadata=part.provider_metadata,
                    )
                )
            elif isinstance(part, messages_.BuiltinToolReturnPart):
                results.append(
                    messages_.ToolResultPart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        result=part.result,
                        result_kind="error" if part.is_error else "json",
                    )
                )
            else:
                parts.append(part)
        if parts:
            out.append(_split(message, parts, first=first))
        if results:
            out.append(messages_.Message(role="tool", parts=results))
    return out


def _split(
    message: messages_.Message, parts: list[messages_.Part], *, first: bool
) -> messages_.Message:
    """One piece of a message: the first keeps its id, the rest get new ones."""
    if first:
        return message.model_copy(update={"parts": parts})
    return messages_.Message(
        role=message.role,
        parts=parts,
        provider_metadata=message.provider_metadata,
    )


def model_view(stored: list[messages_.Message]) -> list[messages_.Message]:
    """Return a stored conversation the way a worker shows it.

    The reverse of `harness_view`: the CLI's tool calls become built-in
    parts, and each result joins the assistant message of its call.
    """
    out: list[messages_.Message] = []
    owner: dict[str, int] = {}  # tool_call_id -> index in `out`
    for message in stored:
        if message.role == "assistant":
            parts: list[messages_.Part] = []
            for part in message.parts:
                if isinstance(part, messages_.ToolCallPart):
                    owner[part.tool_call_id] = len(out)
                    parts.append(
                        messages_.BuiltinToolCallPart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            tool_args=part.tool_args,
                            provider_metadata=part.provider_metadata,
                        )
                    )
                else:
                    parts.append(part)
            out.append(message.model_copy(update={"parts": parts}))
        elif message.role == "tool":
            for result in message.tool_results:
                index = owner.get(result.tool_call_id)
                if index is None:
                    continue
                target = out[index]
                out[index] = target.model_copy(
                    update={
                        "parts": [
                            *target.parts,
                            messages_.BuiltinToolReturnPart(
                                tool_call_id=result.tool_call_id,
                                tool_name=result.tool_name
                                or _name_of(target, result.tool_call_id),
                                result=result.result,
                                is_error=result.result_kind == "error",
                            ),
                        ]
                    }
                )
        else:
            out.append(message)
    return out


def _name_of(message: messages_.Message, tool_call_id: str) -> str:
    return next(
        (
            p.tool_name
            for p in message.parts
            if isinstance(p, messages_.BuiltinToolCallPart)
            and p.tool_call_id == tool_call_id
        ),
        "",
    )


def warn_unsupported_files(
    messages: list[messages_.Message], kind: str
) -> None:
    """Say how many files a history held that `kind` was not given."""
    left = unsupported_files(messages, kind)
    if left:
        warnings.warn(
            f"{left} file(s) in the history were not carried into {kind}: "
            + "it takes images"
            + (" and PDFs" if kind == "claude-code" else "")
            + " from a history. Text, tool calls and images were carried.",
            stacklevel=3,
        )


def unsupported_files(messages: list[messages_.Message], kind: str) -> int:
    """Count files a history holds that `kind` cannot take as input."""
    return sum(
        1
        for m in messages
        for p in m.parts
        if isinstance(p, messages_.FilePart)
        and not p.media_type.startswith("image/")
        and not (kind == "claude-code" and p.media_type == "application/pdf")
    )


def reasoning_count(messages: list[messages_.Message]) -> int:
    return sum(
        1
        for m in messages
        for p in m.parts
        if isinstance(p, messages_.ReasoningPart) and p.text.strip()
    )
