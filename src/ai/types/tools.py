"""Tool schema types — what the LLM layer sees."""

from typing import Any

import pydantic


class ToolSchema(pydantic.BaseModel):
    """What the LLM sees: name, description, and JSON Schema for parameters."""

    name: str
    description: str
    param_schema: dict[str, Any]
    return_type: Any


class BuiltinTool(pydantic.BaseModel):
    """Base for provider-executed built-in tools."""

    model_config = pydantic.ConfigDict(frozen=True)
