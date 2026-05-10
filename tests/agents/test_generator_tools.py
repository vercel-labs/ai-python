"""Async-generator tools and nested streaming via yield_from."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import ai
from ai import models
from ai.types import events as agent_events_
from ai.types import events as events_
from ai.types import messages as messages_

from ..conftest import (
    MOCK_MODEL,
    emit_events_for_messages,
    mock_llm,
    text_msg,
    tool_call_msg,
)

# ---------------------------------------------------------------------------
# Generator tool: yields intermediate messages, returns final text
# ---------------------------------------------------------------------------


@ai.tool(aggregator=ai.LastAggregator)
async def progress_tool(query: str) -> AsyncGenerator[str]:
    """Tool that streams progress, then returns a final answer."""
    yield "Working..."
    yield f"Answer for {query}"


async def test_generator_tool_streams_and_returns_result() -> None:
    """Generator tool yields streaming events visible to consumer;
    final text becomes the tool result fed back to the LLM."""
    my_agent = ai.agent(tools=[progress_tool])

    # Turn 1: LLM calls progress_tool
    # Turn 2: LLM produces final text after seeing the tool result
    call = [tool_call_msg(tc_id="tc-1", name="progress_tool", args='{"query": "test"}')]
    reply = [text_msg("Done!", id="msg-2")]
    llm = mock_llm([call, reply])

    all_events: list[agent_events_.AgentEvent] = []
    async with my_agent.run(MOCK_MODEL, [ai.user_message("Go")]) as stream:
        async for event in stream:
            all_events.append(event)

    assert llm.call_count == 2

    # Intermediate progress events were forwarded to consumer, wrapped
    # in PartialToolCallResult and attributed to the originating tool call.
    progress_wrappers = [
        e for e in all_events if isinstance(e, agent_events_.PartialToolCallResult)
    ]
    assert len(progress_wrappers) == 2
    assert progress_wrappers[0].value == "Working..."

    # Tool result was fed back to LLM.
    tool_results = [
        e for e in all_events if isinstance(e, agent_events_.ToolCallResult)
    ]
    assert len(tool_results) >= 1
    assert tool_results[0].results[0].result == "Answer for test"


# ---------------------------------------------------------------------------
# yield_from: nested agent streams through outer agent
# ---------------------------------------------------------------------------


class _CapturingAdapter:
    """Like MockAdapter but records the messages arg on every call."""

    def __init__(self, responses: list[list[messages_.Message]]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0
        self.calls: list[list[messages_.Message]] = []

    async def stream(
        self,
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        if self._idx >= len(self._responses):
            raise RuntimeError("_CapturingAdapter: no more responses")
        self.call_count += 1
        self.calls.append(list(messages))
        seq = self._responses[self._idx]
        self._idx += 1

        async for event in emit_events_for_messages(seq):
            yield event


@ai.tool
async def inner_fact(topic: str) -> str:
    """Return a fact about a topic."""
    return f"Fact about {topic}"


@ai.tool(aggregator=ai.MessageAggregator)
async def research_tool(topic: str) -> AsyncGenerator[agent_events_.AgentEvent]:
    """Nested agent that researches a topic."""
    inner = ai.agent(tools=[inner_fact])

    msgs = [
        ai.system_message("Be concise."),
        ai.user_message(f"Research: {topic}"),
    ]
    async with inner.run(MOCK_MODEL, msgs) as stream:
        async for event in stream:
            yield event


async def test_yield_from_nested_agent() -> None:
    """yield_from forwards inner events to the consumer but does NOT
    add them to the outer agent's history (context.messages).

    The critical contract: yield_from streams events through the runtime
    queue and discards bare Messages, so the parent agent's
    context.messages stays clean.
    """
    outer = ai.agent(tools=[research_tool])

    # Inner agent: text-only reply (no tools, to keep it simple)
    inner_reply = [text_msg("Mars has two moons.", id="inner-msg")]

    # Outer agent turn 1: calls research_tool
    outer_call = [
        tool_call_msg(tc_id="otc-1", name="research_tool", args='{"topic": "mars"}')
    ]
    # Outer agent turn 2: final answer (after seeing tool result)
    outer_reply = [text_msg("Summary: Mars has two moons.", id="outer-msg-2")]

    adapter = _CapturingAdapter([outer_call, inner_reply, outer_reply])
    models.register_stream("mock", adapter.stream)

    all_events: list[agent_events_.AgentEvent] = []
    async with outer.run(MOCK_MODEL, [ai.user_message("Tell me about Mars")]) as stream:
        async for event in stream:
            all_events.append(event)

    assert adapter.call_count == 3

    # Inner text events were forwarded to the consumer, wrapped in
    # PartialToolCallResult and attributed to the outer tool call.
    inner_text = [
        e
        for e in all_events
        if isinstance(e, agent_events_.PartialToolCallResult)
        and isinstance(e.value, events_.TextDelta)
        and e.value.chunk == "Mars has two moons."
    ]
    assert len(inner_text) > 0
    assert inner_text[0].tool_call_id == "otc-1"
    assert inner_text[0].tool_name == "research_tool"

    tool_results = [
        e for e in all_events if isinstance(e, agent_events_.ToolCallResult)
    ]
    assert tool_results[0].results[0].result == "Mars has two moons."

    # The outer LLM's second call (index 2) must NOT contain any inner
    # agent messages.  It should only see: the original user message,
    # the outer assistant tool-call, and the outer tool-result.
    outer_turn2_msgs = adapter.calls[2]
    outer_turn2_roles = [m.role for m in outer_turn2_msgs]
    assert outer_turn2_roles == ["user", "assistant", "tool"]

    # Specifically: no inner assistant text or inner tool results leaked.
    for m in outer_turn2_msgs:
        if m.role == "assistant":
            # This must be the outer tool-call message, not inner text.
            assert len(m.tool_calls) > 0
