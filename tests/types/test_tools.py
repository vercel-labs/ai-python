"""Focused tests for tool model serialization."""

from __future__ import annotations

from ai.types import tools


def test_function_tool_args_round_trip_through_json_dump() -> None:
    tool = tools.Tool(
        kind="function",
        name="weather",
        args=tools.FunctionToolArgs(
            description="Get weather",
            params={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        ),
    )

    data = tool.model_dump(mode="json")
    restored = tools.Tool.model_validate(data)

    assert data["args"] == {
        "description": "Get weather",
        "params": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    }
    assert isinstance(restored.args, tools.FunctionToolArgs)
    assert restored.args.description == "Get weather"
