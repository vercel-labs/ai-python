"""Checkpoint data model for durable agent execution.

A Checkpoint is a serializable snapshot of all completed work in an agent
run.  On re-entry, the durability provider replays cached results from the
checkpoint instead of re-executing LLM calls and tool invocations.
"""

from __future__ import annotations

from typing import Any

import pydantic

from ..types import messages as messages_


class StepEvent(pydantic.BaseModel):
    """A completed LLM stream step — stores the final done message."""

    index: int
    message: messages_.Message


class ToolEvent(pydantic.BaseModel):
    """A completed tool execution."""

    tool_call_id: str
    tool_name: str
    result: Any
    status: str = "result"  # "result" | "error"


class Checkpoint(pydantic.BaseModel):
    """Serializable snapshot of all completed work in an agent run."""

    steps: list[StepEvent] = []
    tools: list[ToolEvent] = []
