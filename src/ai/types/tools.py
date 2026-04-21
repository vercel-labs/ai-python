"""Tool schema types — what the LLM layer sees.

These are schema-only definitions used by LanguageModel.stream(tools=...).
The executable Tool class and @tool decorator live in agents.agent.
"""

from __future__ import annotations

from typing import Any

import pydantic


class ToolSchema(pydantic.BaseModel):
    """What the LLM sees: name, description, and JSON Schema for parameters."""

    name: str
    description: str
    param_schema: dict[str, Any]
    return_type: Any
