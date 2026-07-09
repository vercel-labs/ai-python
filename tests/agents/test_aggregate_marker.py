"""Aggregate marker — declaring an aggregator via the return-type annotation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

import pytest

import ai
from ai import events as agent_events_

from ..conftest import (
    MOCK_MODEL,
    mock_llm,
    text_msg,
    tool_call_msg,
)


def _factory(t: ai.AgentTool) -> object:
    factory = t.aggregator
    assert factory is not None
    return factory()


def test_tool_return_type_can_be_overridden() -> None:
    @ai.tool(return_type=dict[str, int])
    async def t() -> object:
        return {"x": 1}

    assert t.return_type == dict[str, int]


def test_aggregator_return_type_can_be_overridden() -> None:
    @ai.tool(aggregator=ai.agents.LastAggregator, return_type=int)
    async def t() -> AsyncGenerator[str]:
        yield "1"

    assert t.return_type is int


def test_aggregator_result_type_extracted_from_status_tool_alias() -> None:
    @ai.tool
    async def t() -> ai.StreamingStatusTool[str]:
        yield "x"

    assert t.return_type == str | None


def test_aggregator_result_type_extracted_from_streaming_text_alias() -> None:
    @ai.tool
    async def t() -> ai.StreamingTextTool:
        yield "hello"

    assert t.return_type is str


def test_aggregator_result_type_extracted_from_sub_agent_alias() -> None:
    @ai.tool
    async def t() -> ai.SubAgentTool:
        yield ai.events.StreamStart()

    assert t.return_type is ai.messages.MessageBundle


def test_aggregator_result_type_binds_from_item_to_result() -> None:
    class Box[T](ai.events.Aggregator[T, list[T], str]):
        def __init__(self) -> None:
            self.items: list[T] = []

        def feed(self, item: T) -> None:
            self.items.append(item)

        def snapshot(self) -> list[T]:
            return self.items

        @classmethod
        def to_model_input(cls, snapshot: list[T]) -> str:
            return ""

    @ai.tool(aggregator=Box)
    async def t() -> AsyncGenerator[int]:
        yield 1

    assert t.return_type == list[int]


def test_aggregate_marker_extracted_from_direct_annotated() -> None:
    """Bare ``Annotated[..., Aggregate(...)]`` on the return type."""

    @ai.tool
    async def t() -> (
        Annotated[
            AsyncGenerator[str], ai.agents.Aggregate(ai.agents.LastAggregator)
        ]
    ):
        yield "x"

    assert isinstance(_factory(t), ai.agents.LastAggregator)


def test_aggregate_marker_extracted_from_alias() -> None:
    """``StreamingStatusTool[T]`` carries the marker through a generic alias."""

    @ai.tool
    async def t() -> ai.StreamingStatusTool[str]:
        yield "x"

    assert isinstance(_factory(t), ai.agents.LastAggregator)


def test_sub_agent_tool_alias_extracted() -> None:
    """``SubAgentTool`` (bare alias) carries MessageAggregator."""

    @ai.tool
    async def t() -> ai.SubAgentTool:
        yield ai.events.StreamStart()

    assert isinstance(_factory(t), ai.agents.MessageAggregator)


def test_streaming_text_tool_alias_extracted() -> None:
    """``StreamingTextTool`` (bare alias) carries ConcatAggregator."""

    @ai.tool
    async def t() -> ai.StreamingTextTool:
        yield "hello"
        yield "world"

    agg = _factory(t)
    assert isinstance(agg, ai.agents.ConcatAggregator)
    agg.feed("hello")
    agg.feed("world")
    assert agg.snapshot() == "helloworld"


def test_aggregate_kwarg_passed_to_factory() -> None:
    """Extra kwargs on Aggregate flow through to the factory."""

    @ai.tool
    async def t() -> (
        Annotated[
            AsyncGenerator[str],
            ai.agents.Aggregate(ai.agents.ConcatAggregator, delim="|"),
        ]
    ):
        yield "a"
        yield "b"

    agg = _factory(t)
    assert isinstance(agg, ai.agents.ConcatAggregator)
    agg.feed("a")
    agg.feed("b")
    assert agg.snapshot() == "a|b"


def test_kwarg_and_marker_conflict_raises() -> None:
    """Specifying both ``aggregator=`` and an Aggregate marker is an error."""
    with pytest.raises(TypeError, match="aggregator"):

        @ai.tool(aggregator=ai.agents.LastAggregator)
        async def t() -> ai.StreamingStatusTool[str]:
            yield "x"


def test_multiple_aggregate_markers_raise() -> None:
    """More than one Aggregate marker in the metadata is rejected."""
    with pytest.raises(TypeError, match="multiple Aggregate markers"):

        @ai.tool
        async def t() -> (
            Annotated[
                AsyncGenerator[str],
                ai.agents.Aggregate(ai.agents.LastAggregator),
                ai.agents.Aggregate(ai.agents.ConcatAggregator),
            ]
        ):
            yield "x"


@ai.tool
async def alias_progress_tool(query: str) -> ai.StreamingStatusTool[str]:
    """Smoke test: alias-declared tool runs end-to-end through an agent."""
    yield "Working..."
    yield f"Answer for {query}"


async def test_alias_declared_tool_runs_end_to_end() -> None:
    """An alias-declared streaming tool behaves like the kwarg form."""
    my_agent = ai.Agent(tools=[alias_progress_tool])

    call = [
        tool_call_msg(
            tc_id="tc-1", name="alias_progress_tool", args='{"query": "test"}'
        )
    ]
    reply = [text_msg("Done!", id="msg-2")]
    llm = mock_llm([call, reply])

    all_events: list[agent_events_.AgentEvent] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("Go")]) as stream:
        async for event in stream:
            all_events.append(event)

    assert llm.call_count == 2

    progress = [
        e
        for e in all_events
        if isinstance(e, agent_events_.PartialToolCallResult)
    ]
    assert [p.value for p in progress] == ["Working...", "Answer for test"]

    tool_results = [
        e for e in all_events if isinstance(e, agent_events_.ToolCallResult)
    ]
    tr = tool_results[0].results[0].result
    assert tr == "Answer for test"
