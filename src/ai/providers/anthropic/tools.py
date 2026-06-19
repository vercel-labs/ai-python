"""Anthropic provider-executed tools."""

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

# Beta request headers per provider tool id, merged into the
# ``anthropic-beta`` header by the adapter.
BETA_HEADERS: dict[str, str] = {
    "anthropic.web_search_20260209": "code-execution-web-tools-2026-02-09",
    "anthropic.web_fetch_20260209": "code-execution-web-tools-2026-02-09",
    "anthropic.computer_20251124": "computer-use-2025-11-24",
    "anthropic.bash_20250124": "computer-use-2025-01-24",
    "anthropic.memory_20250818": "context-management-2025-06-27",
}


class UserLocation(pydantic.BaseModel):
    """Approximate user location for geographically relevant search results."""

    model_config = _CONFIG_MODEL

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    region: str | None = None
    country: str | None = None
    timezone: str | None = None


class Citations(pydantic.BaseModel):
    """Citation configuration for web fetch."""

    model_config = _CONFIG_MODEL

    enabled: bool


def _check_domains(
    tool_name: str,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> None:
    if allowed_domains and blocked_domains:
        raise ValueError(
            f"anthropic.{tool_name}: pass only one of "
            "`allowed_domains` or `blocked_domains`"
        )


def _dict_filter_none(**args: Any) -> dict[str, Any]:
    return {k: v for k, v in args.items() if v is not None}


def web_search(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: UserLocation | None = None,
) -> types.tools.Tool:
    _check_domains("web_search", allowed_domains, blocked_domains)
    return types.tools.Tool(
        kind="provider",
        name="web_search",
        tool_config=types.tools.ToolConfig(
            id="anthropic.web_search_20260209",
            args=_dict_filter_none(
                max_uses=max_uses,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                user_location=user_location.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                if user_location is not None
                else None,
            ),
        ),
    )


def web_fetch(
    *,
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    citations: Citations | bool | None = None,
    max_content_tokens: int | None = None,
) -> types.tools.Tool:
    _check_domains("web_fetch", allowed_domains, blocked_domains)
    if isinstance(citations, bool):
        citations = Citations(enabled=citations)
    return types.tools.Tool(
        kind="provider",
        name="web_fetch",
        tool_config=types.tools.ToolConfig(
            id="anthropic.web_fetch_20260209",
            args=_dict_filter_none(
                max_uses=max_uses,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                citations=citations.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                if citations is not None
                else None,
                max_content_tokens=max_content_tokens,
            ),
        ),
    )


def code_execution() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="code_execution",
        tool_config=types.tools.ToolConfig(
            id="anthropic.code_execution_20260120"
        ),
    )


def computer_use(
    *,
    display_width_px: int,
    display_height_px: int,
    display_number: int | None = None,
    enable_zoom: bool | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="computer",
        tool_config=types.tools.ToolConfig(
            id="anthropic.computer_20251124",
            args=_dict_filter_none(
                display_width_px=display_width_px,
                display_height_px=display_height_px,
                display_number=display_number,
                enable_zoom=enable_zoom,
            ),
        ),
    )


def text_editor(*, max_characters: int | None = None) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="str_replace_based_edit_tool",
        tool_config=types.tools.ToolConfig(
            id="anthropic.text_editor_20250728",
            args=_dict_filter_none(max_characters=max_characters),
        ),
    )


def bash() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="bash",
        tool_config=types.tools.ToolConfig(id="anthropic.bash_20250124"),
    )


def memory() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="memory",
        tool_config=types.tools.ToolConfig(id="anthropic.memory_20250818"),
    )


__all__ = [
    "BETA_HEADERS",
    "Citations",
    "UserLocation",
    "bash",
    "code_execution",
    "computer_use",
    "memory",
    "text_editor",
    "web_fetch",
    "web_search",
]
