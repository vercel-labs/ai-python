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
