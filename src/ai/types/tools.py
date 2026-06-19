"""Tool types — what the LLM layer sees."""

from typing import Any, Literal, Self

import pydantic


class ToolApproval(pydantic.BaseModel):
    """Payload schema for tool-approval hooks."""

    granted: bool
    reason: str | None = None


class ToolSpec(pydantic.BaseModel):
    """Model-facing declaration of a host-executed function tool."""

    description: str | None = None
    params: dict[str, Any]


class ToolConfig(pydantic.BaseModel):
    """Execution configuration for a tool.

    For provider-executed tools ``id`` is the canonical provider tool id
    (e.g. ``"anthropic.web_search_20260209"``, ``"openai.mcp"``) and
    ``args`` holds the provider wire arguments as plain snake_case data.
    """

    id: str | None = None
    args: dict[str, Any] = pydantic.Field(default_factory=dict)


class Tool(pydantic.BaseModel):
    kind: Literal["function", "provider"]
    name: str
    spec: ToolSpec | None = None
    tool_config: ToolConfig | None = None
    require_approval: bool = False

    @pydantic.model_validator(mode="after")
    def validate_shape(self) -> Self:
        match self.kind:
            case "function":
                if self.spec is None:
                    raise ValueError(
                        "function tools require spec=ToolSpec(...)"
                    )

            case "provider":
                if self.spec is not None:
                    raise ValueError("provider tools cannot have spec")
                if self.tool_config is None or self.tool_config.id is None:
                    raise ValueError(
                        "provider tools require tool_config with an id"
                    )

        return self
