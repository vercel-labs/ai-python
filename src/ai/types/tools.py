"""Tool types — what the LLM layer sees."""

from typing import Any, Literal, Self

import pydantic


class ToolApproval(pydantic.BaseModel):
    """Payload schema for tool-approval hooks."""

    granted: bool
    reason: str | None = None


class FunctionToolArgs(pydantic.BaseModel):
    description: str | None = pydantic.Field(default=None)
    params: dict[str, Any]


class Tool(pydantic.BaseModel):
    kind: Literal["function", "provider"]
    name: str
    args: pydantic.BaseModel
    require_approval: bool = False

    @pydantic.model_validator(mode="after")
    def validate_args_shape(self) -> Self:
        match self.kind:
            case "function":
                if not isinstance(self.args, FunctionToolArgs):
                    raise ValueError(
                        "function tools require args=FunctionToolArgs(...)"
                    )

            case "provider":
                if isinstance(self.args, FunctionToolArgs):
                    raise ValueError(
                        "provider tools cannot use FunctionToolArgs"
                    )

        return self
