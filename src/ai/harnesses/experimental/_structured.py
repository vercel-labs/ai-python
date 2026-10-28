"""Structured output, as a layer.

Deliberately outside the turn driver. It only calls `run()`, so the settling
path never learns what a schema is — which is what keeps cancellation,
deadlines, and validation from interacting.

It also means the SDK is EDITING the caller's prompt, so `annotate` is the
one place that happens and `Result.prompt_sent` always shows the outcome.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import TypeAdapter

T = TypeVar("T")

INSTRUCTION = (
    "\n\nRespond with ONLY a JSON value matching this schema — "
    + "no prose, no code fences:\n"
)


def annotate(prompt: str, output_type: type[T]) -> str:
    schema = json.dumps(TypeAdapter(output_type).json_schema())
    return prompt + INSTRUCTION + schema


def parse(text: str, output_type: type[T]) -> tuple[T | None, Exception | None]:
    adapter: TypeAdapter[Any] = TypeAdapter(output_type)
    try:
        return adapter.validate_json(_extract_json(text)), None
    except Exception as exc:
        return None, exc


def _extract_json(text: str) -> str:
    """Best effort: the model was told "only JSON", but fences happen."""
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate = parts[1].removeprefix("json").strip()
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if 0 <= start < end:
            sliced = candidate[start : end + 1]
            try:
                json.loads(sliced)
                return sliced
            except json.JSONDecodeError:
                continue
    return candidate
