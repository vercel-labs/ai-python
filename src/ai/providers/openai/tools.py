"""OpenAI provider-executed tools."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pydantic
from pydantic.alias_generators import to_camel

from ... import types

_CONFIG_MODEL = pydantic.ConfigDict(
    frozen=True,
    populate_by_name=True,
    alias_generator=to_camel,
)


class WebSearchUserLocation(pydantic.BaseModel):
    """User-location hint for OpenAI web search."""

    model_config = _CONFIG_MODEL

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class WebSearchFilters(pydantic.BaseModel):
    """Filters for OpenAI web search."""

    model_config = _CONFIG_MODEL

    allowed_domains: list[str] | None = None


class FileSearchRanking(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    ranker: str | None = None
    score_threshold: float | None = None


class CodeInterpreterContainer(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    type: Literal["auto"] = "auto"
    file_ids: list[str] | None = None


class OpenAIProviderArgs(pydantic.BaseModel):
    """Base for OpenAI provider-executed tool args."""

    model_config = _CONFIG_MODEL

    openai_id: ClassVar[str]


class WebSearchArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.web_search"

    model_config = _CONFIG_MODEL

    external_web_access: bool | None = None
    filters: WebSearchFilters | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None


class WebSearchPreviewArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.web_search_preview"

    model_config = _CONFIG_MODEL

    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: WebSearchUserLocation | None = None


class FileSearchArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.file_search"

    model_config = _CONFIG_MODEL

    vector_store_ids: list[str]
    max_num_results: int | None = None
    ranking: FileSearchRanking | None = None
    filters: dict[str, Any] | None = None


class CodeInterpreterArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.code_interpreter"

    model_config = _CONFIG_MODEL

    container: CodeInterpreterContainer | str | None = None


class ImageGenerationArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.image_generation"

    model_config = _CONFIG_MODEL

    background: Literal["transparent", "opaque", "auto"] | None = None
    input_fidelity: Literal["high", "low"] | None = None
    model: str | None = None
    moderation: Literal["auto", "low"] | None = None
    output_compression: int | None = None
    output_format: Literal["png", "webp", "jpeg"] | None = None
    partial_images: int | None = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    size: str | None = None


class LocalShellArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.local_shell"

    model_config = _CONFIG_MODEL


class ShellArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.shell"

    model_config = _CONFIG_MODEL

    environment: str | None = None


class ApplyPatchArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.apply_patch"

    model_config = _CONFIG_MODEL


class McpArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.mcp"

    model_config = _CONFIG_MODEL

    server_label: str
    server_url: str | None = None
    connector_id: str | None = None
    authorization: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: list[str] | dict[str, Any] | None = None
    server_description: str | None = None


class ToolSearchArgs(OpenAIProviderArgs):
    openai_id: ClassVar[str] = "openai.tool_search"

    model_config = _CONFIG_MODEL

    description: str | None = None
    parameters: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None


def web_search(
    *,
    external_web_access: bool | None = None,
    filters: WebSearchFilters | None = None,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="web_search",
        args=WebSearchArgs(
            external_web_access=external_web_access,
            filters=filters,
            search_context_size=search_context_size,
            user_location=user_location,
        ),
    )


def web_search_preview(
    *,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="web_search_preview",
        args=WebSearchPreviewArgs(
            search_context_size=search_context_size,
            user_location=user_location,
        ),
    )


def file_search(
    *,
    vector_store_ids: list[str],
    max_num_results: int | None = None,
    ranking: FileSearchRanking | None = None,
    filters: dict[str, Any] | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="file_search",
        args=FileSearchArgs(
            vector_store_ids=vector_store_ids,
            max_num_results=max_num_results,
            ranking=ranking,
            filters=filters,
        ),
    )


def code_interpreter(
    *,
    container: CodeInterpreterContainer | str | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="code_interpreter",
        args=CodeInterpreterArgs(container=container),
    )


def image_generation(
    *,
    background: Literal["transparent", "opaque", "auto"] | None = None,
    input_fidelity: Literal["high", "low"] | None = None,
    model: str | None = None,
    moderation: Literal["auto", "low"] | None = None,
    output_compression: int | None = None,
    output_format: Literal["png", "webp", "jpeg"] | None = None,
    partial_images: int | None = None,
    quality: Literal["low", "medium", "high", "auto"] | None = None,
    size: str | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="image_generation",
        args=ImageGenerationArgs(
            background=background,
            input_fidelity=input_fidelity,
            model=model,
            moderation=moderation,
            output_compression=output_compression,
            output_format=output_format,
            partial_images=partial_images,
            quality=quality,
            size=size,
        ),
    )


def local_shell() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider", name="local_shell", args=LocalShellArgs()
    )


def shell(*, environment: str | None = None) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="shell",
        args=ShellArgs(environment=environment),
    )


def apply_patch() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="apply_patch",
        args=ApplyPatchArgs(),
    )


def mcp(
    *,
    server_label: str,
    server_url: str | None = None,
    connector_id: str | None = None,
    authorization: str | None = None,
    headers: dict[str, str] | None = None,
    allowed_tools: list[str] | dict[str, Any] | None = None,
    server_description: str | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="mcp",
        args=McpArgs(
            server_label=server_label,
            server_url=server_url,
            connector_id=connector_id,
            authorization=authorization,
            headers=headers,
            allowed_tools=allowed_tools,
            server_description=server_description,
        ),
    )


def tool_search(
    *,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="tool_search",
        args=ToolSearchArgs(
            description=description,
            parameters=parameters,
            execution=execution,
        ),
    )


__all__ = [
    "ApplyPatchArgs",
    "CodeInterpreterArgs",
    "CodeInterpreterContainer",
    "FileSearchArgs",
    "FileSearchRanking",
    "ImageGenerationArgs",
    "LocalShellArgs",
    "McpArgs",
    "OpenAIProviderArgs",
    "ShellArgs",
    "ToolSearchArgs",
    "WebSearchArgs",
    "WebSearchFilters",
    "WebSearchPreviewArgs",
    "WebSearchUserLocation",
    "apply_patch",
    "code_interpreter",
    "file_search",
    "image_generation",
    "local_shell",
    "mcp",
    "shell",
    "tool_search",
    "web_search",
    "web_search_preview",
]
