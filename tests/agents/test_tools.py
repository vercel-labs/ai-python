"""@tool decorator: schema extraction, execution, ToolCall."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Annotated, Any, cast

import pydantic
import pytest

import ai
from ai.types import events as events_

# Declaring a tool with a plain `def` is a type error by design; go
# through an untyped view of the decorator to reach the runtime check.
_untyped_tool = cast(
    "Callable[[Callable[..., Any]], ai.AgentTool]",
    ai.tool,
)

# Combining an aggregator with to_model_input is a type error by
# design; reach the runtime check the same way.
_untyped_tool_kw = cast(
    "Callable[..., Callable[[Callable[..., Any]], ai.AgentTool]]",
    ai.tool,
)

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
    assert not result.results[0].has_model_input


# -- ToolCall binds a ToolCallPart to a Tool and returns tool messages ----


async def test_tool_runner_discard_cancels_and_omits_task() -> None:
    """Discarded speculative calls do not produce tool events or messages."""
    started = asyncio.Event()

    async def speculative() -> events_.ToolCallResult:
        started.set()
        await asyncio.Future()
        return ai.tool_result(tool_call_id="tc-speculative", result="unused")

    async with ai.ToolRunner() as runner:
        task = runner.schedule(speculative)
        assert isinstance(task, asyncio.Task)
        await started.wait()
        runner.discard(task)

        assert [event async for event in runner.events()] == []
        assert runner.get_tool_message() is None
        with pytest.raises(asyncio.CancelledError):
            await task


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
    assert not result.results[0].has_model_input


async def test_cancelled_tool_call_returns_error_result() -> None:
    started = asyncio.Event()

    @ai.tool
    async def wait_forever() -> str:
        """Wait until cancelled."""
        started.set()
        await asyncio.Future()
        return "unreachable"

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-cancelled",
        tool_name="wait_forever",
        tool_args="{}",
    )
    task = asyncio.create_task(
        ai.agents.BoundToolCall(part=part, tool=wait_forever)()
    )
    await started.wait()
    task.cancel()
    result = await task

    assert result.results[0].is_error
    assert "CancelledError" in str(result.results[0].result)
    assert isinstance(result.exception, asyncio.CancelledError)


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


async def test_sync_tool_says_what_is_wrong() -> None:
    # Typing rejects a plain `def` tool; if one gets through anyway, the
    # failure names the tool instead of "object int can't be used in
    # 'await' expression".
    @_untyped_tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-sync",
        tool_name="add",
        tool_args='{"a": 1, "b": 2}',
    )
    result = await ai.agents.BoundToolCall(part=part, tool=add)()

    assert result.results[0].is_error
    assert "must return an awaitable" in str(result.results[0].result)
    assert "'add'" in str(result.results[0].result)
    assert isinstance(result.exception, TypeError)


async def test_plain_def_returning_a_coroutine_runs() -> None:
    # The check is on the returned value, not on how the tool is
    # declared, so a `def` that hands back a coroutine is fine.
    async def _add(a: int, b: int) -> int:
        return a + b

    @_untyped_tool
    def add(a: int, b: int) -> Any:
        """Add two numbers."""
        return _add(a, b)

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-coro",
        tool_name="add",
        tool_args='{"a": 1, "b": 2}',
    )
    result = await ai.agents.BoundToolCall(part=part, tool=add)()

    assert not result.results[0].is_error
    assert result.results[0].result == 3


async def test_generator_tool_without_aggregator_says_so() -> None:
    @_untyped_tool
    async def stream(x: int) -> Any:
        """Streams without declaring an aggregator."""
        yield x

    part = ai.messages.ToolCallPart(
        tool_call_id="tc-gen",
        tool_name="stream",
        tool_args='{"x": 1}',
    )
    result = await ai.agents.BoundToolCall(part=part, tool=stream)()

    assert result.results[0].is_error
    assert "declares no aggregator" in str(result.results[0].result)


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


# -- to_model_input -------------------------------------------------------


async def _run_tool(tool: ai.AgentTool, args: str = "{}") -> Any:
    part = ai.messages.ToolCallPart(
        tool_call_id=f"tc-{tool.name}",
        tool_name=tool.name,
        tool_args=args,
    )
    result = await ai.agents.BoundToolCall(part=part, tool=tool)()
    return result.results[0]


class _EditResult(pydantic.BaseModel):
    message: str
    old_content: str
    new_content: str


async def test_to_model_input_transforms_what_the_model_sees() -> None:
    """The result keeps the full value; the model sees the conversion."""

    @ai.tool(to_model_input=lambda r: r.message)
    async def edit(path: str) -> _EditResult:
        """Edit a file."""
        return _EditResult(
            message=f"edited {path}", old_content="a", new_content="b"
        )

    part = await _run_tool(edit, '{"path": "f.py"}')

    assert isinstance(part.result, _EditResult)
    assert part.result.old_content == "a"
    assert part.has_model_input
    assert part.get_model_input() == "edited f.py"


async def test_to_model_input_not_called_without_it() -> None:
    @ai.tool
    async def plain() -> str:
        """Plain tool."""
        return "hello"

    part = await _run_tool(plain)

    assert not part.has_model_input
    assert part.get_model_input() == "hello"


async def test_to_model_input_not_called_on_error() -> None:
    calls: list[Any] = []

    @ai.tool(to_model_input=calls.append)
    async def boom() -> str:
        """Always fails."""
        raise RuntimeError("nope")

    part = await _run_tool(boom)

    assert part.is_error
    assert not part.has_model_input
    assert calls == []


async def test_to_model_input_failure_becomes_a_tool_error() -> None:
    def convert(result: str) -> str:
        raise ValueError("bad conversion")

    @ai.tool(to_model_input=convert)
    async def t() -> str:
        """Tool."""
        return "ok"

    part = await _run_tool(t)

    assert part.is_error
    assert "bad conversion" in str(part.result)


def test_to_model_input_with_aggregator_is_rejected() -> None:
    with pytest.raises(TypeError, match="cannot be combined"):

        @_untyped_tool_kw(
            aggregator=ai.agents.LastAggregator,
            to_model_input=str,
        )
        async def t() -> AsyncGenerator[str]:
            """Stream."""
            yield "x"


def test_to_model_input_with_aggregate_marker_is_rejected() -> None:
    """The marker form of the aggregator conflicts too."""
    with pytest.raises(TypeError, match="cannot be combined"):

        @ai.tool(to_model_input=str)
        async def t() -> ai.StreamingTextTool:
            """Stream."""
            yield "x"


# -- Parameter descriptions -------------------------------------------------


def test_annotated_field_description_lands_in_schema() -> None:
    @ai.tool
    async def get_weather(
        city: Annotated[str, pydantic.Field(description="City name")],
        days: Annotated[
            int, pydantic.Field(ge=1, le=7, description="Forecast length")
        ],
    ) -> str:
        """Get the weather."""
        return "sunny"

    props = _schema(get_weather)["properties"]
    assert props["city"]["description"] == "City name"
    assert props["days"]["description"] == "Forecast length"
    assert props["days"]["minimum"] == 1
    assert props["days"]["maximum"] == 7


def test_annotated_inside_optional_is_not_stripped() -> None:
    @ai.tool
    async def search(
        query: str,
        limit: Annotated[int, pydantic.Field(ge=1, description="Max results")]
        | None = None,
    ) -> str:
        """Search."""
        return query

    # For `X | None` unions the metadata lives on the non-null branch.
    prop = _schema(search)["properties"]["limit"]
    branch = next(b for b in prop["anyOf"] if b.get("type") == "integer")
    assert branch["description"] == "Max results"
    assert branch["minimum"] == 1


def test_docstring_args_section_fills_descriptions() -> None:
    @ai.tool
    async def deliver(
        address: str,
        fragile: bool,
        note: str = "",
    ) -> str:
        """Deliver a package.

        Args:
            address: Street address to ship to.
                Can span multiple lines.
            fragile: Whether the contents break easily.
        """
        return address

    props = _schema(deliver)["properties"]
    assert props["address"]["description"] == (
        "Street address to ship to. Can span multiple lines."
    )
    assert (
        props["fragile"]["description"] == "Whether the contents break easily."
    )
    # `note` has no docstring entry and no Field: no description is invented.
    assert "description" not in props["note"]


def test_annotated_description_wins_over_docstring() -> None:
    @ai.tool
    async def park(
        plate: Annotated[
            str, pydantic.Field(description="License plate number")
        ],
    ) -> str:
        """Park a car.

        Args:
            plate: The plate, apparently.
        """
        return plate

    assert (
        _schema(park)["properties"]["plate"]["description"]
        == "License plate number"
    )


def test_docstring_without_args_leaves_schema_untouched() -> None:
    @ai.tool
    async def ping(host: str) -> str:
        """Ping a host. Returns latency in ms.

        Returns:
            Latency string.
        """
        return host

    schema_before = {
        "properties": {"host": {"title": "Host", "type": "string"}},
        "required": ["host"],
        "title": "ping_Args",
        "type": "object",
    }
    assert _schema(ping) == schema_before


def test_docstring_param_names_not_in_signature_are_ignored() -> None:
    @ai.tool
    async def bake(kind: str) -> str:
        """Bake something.

        Args:
            kind: What to bake.
            oven_secret: Not a real parameter.
        """
        return kind

    props = _schema(bake)["properties"]
    assert "oven_secret" not in props
    assert props["kind"]["description"] == "What to bake."


def test_plain_hints_schema_is_byte_identical_to_pre_change_behavior() -> None:
    @ai.tool
    async def greet(name: str, count: int = 1) -> str:
        """Say hello."""
        return "hi" * count

    assert _schema(greet) == {
        "properties": {
            "count": {"default": 1, "title": "Count", "type": "integer"},
            "name": {"title": "Name", "type": "string"},
        },
        "required": ["name"],
        "title": "greet_Args",
        "type": "object",
    }
