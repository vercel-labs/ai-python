from __future__ import annotations

from typing import Annotated, Any, Literal

import pydantic

from ... import types

_METADATA_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class AnthropicCitations(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    enabled: bool


class AnthropicCaller(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    type: str
    tool_id: str | None = pydantic.Field(
        default=None,
        validation_alias="toolId",
        serialization_alias="toolId",
    )


class AnthropicUsageIteration(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    type: Literal["compaction", "message"]
    input_tokens: int = pydantic.Field(
        validation_alias="inputTokens",
        serialization_alias="inputTokens",
    )
    output_tokens: int = pydantic.Field(
        validation_alias="outputTokens",
        serialization_alias="outputTokens",
    )


class AnthropicContainerSkill(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    type: Literal["anthropic", "custom"]
    skill_id: str = pydantic.Field(
        validation_alias="skillId",
        serialization_alias="skillId",
    )
    version: str


class AnthropicContainerMetadata(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    expires_at: str = pydantic.Field(
        validation_alias="expiresAt",
        serialization_alias="expiresAt",
    )
    id: str
    skills: list[AnthropicContainerSkill] | None = None


class AnthropicClearToolUsesEdit(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    type: Literal["clear_tool_uses_20250919"]
    cleared_tool_uses: int = pydantic.Field(
        validation_alias="clearedToolUses",
        serialization_alias="clearedToolUses",
    )
    cleared_input_tokens: int = pydantic.Field(
        validation_alias="clearedInputTokens",
        serialization_alias="clearedInputTokens",
    )


class AnthropicClearThinkingEdit(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    type: Literal["clear_thinking_20251015"]
    cleared_thinking_turns: int = pydantic.Field(
        validation_alias="clearedThinkingTurns",
        serialization_alias="clearedThinkingTurns",
    )
    cleared_input_tokens: int = pydantic.Field(
        validation_alias="clearedInputTokens",
        serialization_alias="clearedInputTokens",
    )


class AnthropicCompactEdit(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    type: Literal["compact_20260112"]


type AnthropicContextManagementEdit = Annotated[
    AnthropicClearToolUsesEdit | AnthropicClearThinkingEdit | AnthropicCompactEdit,
    pydantic.Field(discriminator="type"),
]


class AnthropicContextManagementMetadata(pydantic.BaseModel):
    model_config = _METADATA_CONFIG

    applied_edits: list[AnthropicContextManagementEdit] = pydantic.Field(
        validation_alias="appliedEdits",
        serialization_alias="appliedEdits",
    )


class AnthropicProviderMetadata(types.metadata.ProviderMetadata):
    """Anthropic-specific metadata for messages, parts, and stream events."""

    provider: Literal["anthropic"] = "anthropic"

    # Input/replay options.
    cache_control: dict[str, Any] | None = pydantic.Field(
        default=None,
        validation_alias="cacheControl",
        serialization_alias="cacheControl",
    )
    citations: AnthropicCitations | None = None
    title: str | None = None
    context: str | None = None

    # Special part markers.
    type: Literal["compaction", "mcp-tool-use"] | None = None
    server_name: str | None = pydantic.Field(
        default=None,
        validation_alias="serverName",
        serialization_alias="serverName",
    )

    # Reasoning replay metadata.
    signature: str | None = None
    redacted_data: str | None = pydantic.Field(
        default=None,
        validation_alias="redactedData",
        serialization_alias="redactedData",
    )

    # Tool replay metadata.
    result_type: str | None = pydantic.Field(
        default=None,
        validation_alias="resultType",
        serialization_alias="resultType",
    )
    caller: AnthropicCaller | dict[str, Any] | None = None

    # Response/message metadata.
    usage: dict[str, Any] | None = None
    stop_sequence: str | None = pydantic.Field(
        default=None,
        validation_alias="stopSequence",
        serialization_alias="stopSequence",
    )
    iterations: list[AnthropicUsageIteration] | None = None
    container: AnthropicContainerMetadata | None = None
    context_management: AnthropicContextManagementMetadata | None = pydantic.Field(
        default=None,
        validation_alias="contextManagement",
        serialization_alias="contextManagement",
    )
