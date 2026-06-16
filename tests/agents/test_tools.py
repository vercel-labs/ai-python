"""@tool decorator: schema extraction, execution, ToolCall."""

from __future__ import annotations

import asyncio
from typing import Any, cast

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
    assert _spec(greet).description == "Say hello."
    props = _schema(greet)["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert set(_schema(greet)["required"]) == {"name", "count"}


def test_optional_param_not_required() -> None:
    @ai.tool
    async def search(query: str, limit: int | None = None) -> str:
        """Search."""
        return query

    schema = _schema(search)
    assert "query" in schema.get("required", [])
    assert "limit" not in schema.get("required", [])
    assert "limit" in schema["properties"]


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

    props = _schema(send)["properties"]
    assert props["recipients"]["type"] == "array"
    assert props["recipients"]["items"]["type"] == "string"


# -- Execution (ToolCall) --------------------------------------------------


async def test_tool_call_with_json_args() -> None:
    @ai.tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-add",
        tool_name="add",
        tool_args='{"a": 1, "b": 2}',
    )
    result = await ai.agents.BoundToolCall(part=part, tool=add)()
    out = result.results[0].result
    assert out == 3


# -- ToolCall binds a ToolCallPart to a Tool and returns tool messages ----


async def test_tool_call_returns_tool_message() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.agents.BoundToolCall(part=part, tool=double)
    result = await tc()

    assert cast(Any, tc.fn).__name__ == "double"
    assert tc.kwargs == {"x": 5}
    assert result.message.role == "tool"
    assert len(result.results) == 1
    assert result.results[0].tool_call_id == "tc-1"
    assert result.results[0].tool_name == "double"
    out = result.results[0].result
    assert out == 10
    assert not result.results[0].is_error


async def test_tool_call_catches_errors() -> None:
    @ai.tool
    async def fail(x: int) -> int:
        """Always fails."""
        raise ValueError("boom")

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-err",
        tool_name="fail",
        tool_args='{"x": 1}',
    )
    tc = ai.agents.BoundToolCall(part=part, tool=fail)
    result = await tc()

    assert result.results[0].is_error
    assert "boom" in str(result.results[0].result)
    # The real exception is preserved on the event for richer logging.
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "boom"


async def test_tool_call_unwraps_singleton_exceptiongroup() -> None:
    """When a tool's body raises an ExceptionGroup wrapping a single
    exception (typical when it runs an asyncio TaskGroup internally),
    the auto-catch surfaces the underlying error — not the group."""

    class BoomError(RuntimeError):
        pass

    @ai.tool
    async def fail_via_group(x: int) -> int:
        """Fails inside a TaskGroup."""

        async def _inner() -> None:
            raise BoomError("kaboom")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(_inner())
        return x  # unreachable

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-grp",
        tool_name="fail_via_group",
        tool_args='{"x": 1}',
    )
    tc = ai.agents.BoundToolCall(part=part, tool=fail_via_group)
    result = await tc()

    assert result.results[0].is_error
    # Result text reflects the unwrapped exception type, not BaseExceptionGroup.
    assert "BoomError" in str(result.results[0].result)
    assert "kaboom" in str(result.results[0].result)
    assert isinstance(result.exception, BoomError)


async def test_tool_call_allows_kwarg_overrides() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.agents.BoundToolCall(part=part, tool=double)

    result = await tc(x=7)

    out = result.results[0].result
    assert out == 14


async def test_tool_call_override_validation_failure() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": 5}',
    )
    tc = ai.agents.BoundToolCall(part=part, tool=double)

    with pytest.raises(pydantic.ValidationError):
        await tc(x="bad")


async def test_tool_call_malformed_args_become_error_message() -> None:
    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-1",
        tool_name="double",
        tool_args='{"x": ',
    )
    tc = ai.agents.BoundToolCall(part=part, tool=double)

    result = await tc()

    assert result.results[0].is_error


# -- Helpers ---------------------------------------------------------------


def _required(tool: ai.AgentTool) -> list[str]:
    result = _schema(tool).get("required", [])
    assert isinstance(result, list)
    return result


def _schema(tool: ai.AgentTool) -> dict[str, Any]:
    return _spec(tool).params


def _spec(tool: ai.AgentTool) -> ai.tools.ToolSpec:
    spec = tool.tool.spec
    assert spec is not None
    return spec


# Module-level model so get_type_hints() can resolve it when @ai.tool
# inspects the decorated function's annotations.
class _NestedItem(pydantic.BaseModel):
    key: str
    value: str


async def test_tool_call_with_nested_pydantic_model() -> None:
    """Tools whose parameters include nested Pydantic models receive model
    instances, not plain dicts.  Regression test for the model_dump() bug
    where _validate_kwargs serialised nested models to dicts before passing
    them to the tool function."""

    received: list[_NestedItem] = []

    @ai.tool
    async def store(items: list[_NestedItem]) -> str:
        """Store items."""
        received.extend(items)
        return "ok"

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-nested",
        tool_name="store",
        tool_args=(
            '{"items": [{"key": "a", "value": "1"}, '
            '{"key": "b", "value": "2"}]}'
        ),
    )
    result = await ai.agents.BoundToolCall(part=part, tool=store)()

    assert not result.results[0].is_error
    assert len(received) == 2
    assert all(
        isinstance(item, _NestedItem) for item in received
    ), f"expected _NestedItem instances, got {[type(i) for i in received]}"
    assert received[0].key == "a"
    assert received[1].value == "2"
