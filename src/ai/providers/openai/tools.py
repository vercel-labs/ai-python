"""OpenAI provider-executed tools."""

from __future__ import annotations

from typing import Any, Literal

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


def _provider_tool(name: str, id: str, **args: Any) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name=name,
        tool_config=types.tools.ToolConfig(
            id=id,
            args={k: v for k, v in args.items() if v is not None},
        ),
    )


def _dump(model: pydantic.BaseModel | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return model.model_dump(mode="json", exclude_none=True)


def web_search(
    *,
    external_web_access: bool | None = None,
    filters: WebSearchFilters | None = None,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> types.tools.Tool:
    return _provider_tool(
        "web_search",
        "openai.web_search",
        external_web_access=external_web_access,
        filters=_dump(filters),
        search_context_size=search_context_size,
        user_location=_dump(user_location),
    )


def web_search_preview(
    *,
    search_context_size: Literal["low", "medium", "high"] | None = None,
    user_location: WebSearchUserLocation | None = None,
) -> types.tools.Tool:
    return _provider_tool(
        "web_search_preview",
        "openai.web_search_preview",
        search_context_size=search_context_size,
        user_location=_dump(user_location),
    )


def file_search(
    *,
    vector_store_ids: list[str],
    max_num_results: int | None = None,
    ranking: FileSearchRanking | None = None,
    filters: dict[str, Any] | None = None,
) -> types.tools.Tool:
    return _provider_tool(
        "file_search",
        "openai.file_search",
        vector_store_ids=vector_store_ids,
        max_num_results=max_num_results,
        ranking=_dump(ranking),
        filters=filters,
    )


def code_interpreter(
    *,
    container: CodeInterpreterContainer | str | None = None,
) -> types.tools.Tool:
    return _provider_tool(
        "code_interpreter",
        "openai.code_interpreter",
        container=_dump(container)
        if isinstance(container, CodeInterpreterContainer)
        else container,
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
    return _provider_tool(
        "image_generation",
        "openai.image_generation",
        background=background,
        input_fidelity=input_fidelity,
        model=model,
        moderation=moderation,
        output_compression=output_compression,
        output_format=output_format,
        partial_images=partial_images,
        quality=quality,
        size=size,
    )


def local_shell() -> types.tools.Tool:
    return _provider_tool("local_shell", "openai.local_shell")


def shell(*, environment: str | None = None) -> types.tools.Tool:
    return _provider_tool("shell", "openai.shell", environment=environment)


def apply_patch() -> types.tools.Tool:
    return _provider_tool("apply_patch", "openai.apply_patch")


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
    return _provider_tool(
        "mcp",
        "openai.mcp",
        server_label=server_label,
        server_url=server_url,
        connector_id=connector_id,
        authorization=authorization,
        headers=headers,
        allowed_tools=allowed_tools,
        server_description=server_description,
    )


def tool_search(
    *,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> types.tools.Tool:
    return _provider_tool(
        "tool_search",
        "openai.tool_search",
        description=description,
        parameters=parameters,
        execution=execution,
    )


__all__ = [
    "CodeInterpreterContainer",
    "FileSearchRanking",
    "WebSearchFilters",
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
