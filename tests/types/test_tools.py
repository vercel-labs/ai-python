"""Tool data model: shape validation and JSON roundtripping."""

from __future__ import annotations

import pydantic
import pytest

from ai import types
from ai.providers.anthropic import tools as anthropic_tools
from ai.providers.openai import tools as openai_tools


def _roundtrip(tool: types.tools.Tool) -> types.tools.Tool:
    return types.tools.Tool.model_validate_json(tool.model_dump_json())


def test_function_tool_roundtrips_through_json() -> None:
    tool = types.tools.Tool(
        kind="function",
        name="get_weather",
        spec=types.tools.ToolSpec(
            description="Get the weather.",
            params={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
        require_approval=True,
    )

    assert _roundtrip(tool) == tool


def test_provider_tool_roundtrips_through_json() -> None:
    tool = anthropic_tools.web_search(
        max_uses=3,
        allowed_domains=["example.com"],
        user_location=anthropic_tools.UserLocation(city="SF", country="US"),
    )

    assert _roundtrip(tool) == tool


def test_provider_tool_with_opaque_payload_roundtrips() -> None:
    tool = openai_tools.mcp(
        server_label="my-server",
        headers={"x_api_key": "secret"},
    )

    assert _roundtrip(tool) == tool


def test_function_tool_requires_spec() -> None:
    with pytest.raises(pydantic.ValidationError, match="require spec"):
        types.tools.Tool(kind="function", name="bad")


def test_provider_tool_rejects_spec() -> None:
    with pytest.raises(pydantic.ValidationError, match="cannot have spec"):
        types.tools.Tool(
            kind="provider",
            name="bad",
            spec=types.tools.ToolSpec(params={}),
            tool_config=types.tools.ToolConfig(id="x.y"),
        )


def test_provider_tool_requires_tool_config_id() -> None:
    with pytest.raises(pydantic.ValidationError, match="tool_config"):
        types.tools.Tool(kind="provider", name="bad")

    with pytest.raises(pydantic.ValidationError, match="tool_config"):
        types.tools.Tool(
            kind="provider",
            name="bad",
            tool_config=types.tools.ToolConfig(),
        )
