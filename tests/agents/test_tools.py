"""@tool decorator: schema extraction, execution, ToolCall."""

import pydantic
import pytest

import ai

# -- Schema extraction from type hints ------------------------------------


def test_simple_types_produce_correct_schema() -> None:
    @ai.tool
    async def greet(name: str, count: int) -> str:
        """Say hello."""
        return f"Hello {name}" * count

    assert greet.name == "greet"
    assert greet.description == "Say hello."
    props = greet.param_schema["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert set(greet.param_schema["required"]) == {"name", "count"}


def test_optional_param_not_required() -> None:
    @ai.tool
    async def search(query: str, limit: int | None = None) -> str:
        """Search."""
        return query

    assert "query" in search.param_schema.get("required", [])
    assert "limit" not in search.param_schema.get("required", [])
    assert "limit" in search.param_schema["properties"]


def test_default_value_not_required() -> None:
    @ai.tool
    async def fetch(url: str, timeout: int = 30) -> str:
        """Fetch URL."""
        return url

    assert "url" in _required(fetch)
    assert "timeout" not in _required(fetch)


def test_complex_type_schema() -> None:
    @ai.tool
    async def send(recipients: list[str], urgent: bool = False) -> str:
        """Send message."""
        return "sent"

    props = send.param_schema["properties"]
    assert props["recipients"]["type"] == "array"
    assert props["recipients"]["items"]["type"] == "string"


# -- Execution (Tool.__call__) --------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_with_json_args() -> None:
    @ai.tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = await add('{"a": 1, "b": 2}')
    assert result == 3


# -- ToolCall binds a ToolCallPart to a Tool and returns tool messages ----


@pytest.mark.asyncio
async def test_tool_call_returns_tool_message() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.ToolCall(part=part, tool=double)
    result = await tc()

    assert tc.fn is double.fn
    assert tc.kwargs == {"x": 5}
    assert result.role == "tool"
    assert len(result.tool_results) == 1
    assert result.tool_results[0].tool_call_id == "tc-1"
    assert result.tool_results[0].tool_name == "double"
    assert result.tool_results[0].result == 10
    assert not result.tool_results[0].is_error


@pytest.mark.asyncio
async def test_tool_call_catches_errors() -> None:
    @ai.tool
    async def fail(x: int) -> int:
        """Always fails."""
        raise ValueError("boom")

    part = ai.ToolCallPart(
        tool_call_id="tc-err",
        tool_name="fail",
        tool_args='{"x": 1}',
    )
    tc = ai.ToolCall(part=part, tool=fail)
    result = await tc()

    assert result.tool_results[0].is_error
    assert "boom" in str(result.tool_results[0].result)


@pytest.mark.asyncio
async def test_tool_call_allows_kwarg_overrides() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.ToolCall(part=part, tool=double)

    result = await tc(x=7)

    assert result.tool_results[0].result == 14


@pytest.mark.asyncio
async def test_tool_call_override_validation_failure() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.ToolCall(part=part, tool=double)

    with pytest.raises(pydantic.ValidationError):
        await tc(x="bad")


@pytest.mark.asyncio
async def test_tool_call_malformed_args_become_error_message() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": ',
    )
    tc = ai.ToolCall(part=part, tool=double)

    result = await tc()

    assert result.tool_results[0].is_error


# -- Helpers ---------------------------------------------------------------


def _required(tool: ai.Tool[..., object]) -> list[str]:
    result = tool.param_schema.get("required", [])
    assert isinstance(result, list)
    return result
