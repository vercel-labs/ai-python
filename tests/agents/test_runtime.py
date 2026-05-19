"""Agent default loop, tool execution, multi-turn."""

from __future__ import annotations

import ai
from ai.types import messages

from ..conftest import (
    MOCK_MODEL,
    collect_messages,
    mock_llm,
    text_msg,
    tool_call_msg,
)

# -- Tool definitions for tests --------------------------------------------


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    return x * 2


@ai.tool
async def concat(a: str, b: str) -> str:
    """Concatenate strings."""
    return a + b


# -- Agent default loop: single turn (no tools) ----------------------------


async def test_agent_text_only() -> None:
    """Agent default loop with no tool calls returns after one LLM call."""
    my_agent = ai.agent(tools=[double])

    llm = mock_llm([[text_msg("Hello!")]])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]) as stream:
        msgs = await collect_messages(stream)
    assert llm.call_count == 1
    assert any(m.text == "Hello!" for m in msgs)


# -- Agent default loop: tool call + follow-up -----------------------------


async def test_agent_tool_then_text() -> None:
    """Agent default loop calls tool, feeds result back, gets final text."""
    my_agent = ai.agent(tools=[double])

    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}')]
    call2 = [text_msg("The answer is 10.")]
    llm = mock_llm([call1, call2])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("Double 5")]
    ) as stream:
        msgs = await collect_messages(stream)
    assert llm.call_count == 2
    tool_results = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_results[0].result == 10


# -- Agent default loop: multiple tool calls in one message ----------------


async def test_agent_parallel_tools() -> None:
    """LLM returns two tool calls in one message; both execute."""
    my_agent = ai.agent(tools=[double])

    two_tools = messages.Message(
        id="msg-1",
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="double",
                tool_args='{"x": 3}',
            ),
            messages.ToolCallPart(
                tool_call_id="tc-2",
                tool_name="double",
                tool_args='{"x": 7}',
            ),
        ],
    )
    call2 = [text_msg("6 and 14", id="msg-2")]
    llm = mock_llm([[two_tools], call2])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("Double 3 and 7")]
    ) as stream:
        msgs = await collect_messages(stream)
    assert llm.call_count == 2
    tool_result_msgs = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_result_msgs) >= 1


# -- Agent default loop: multi-turn (tool -> tool -> text) -----------------


async def test_agent_multi_turn() -> None:
    """LLM calls a tool, then calls another tool, then returns text."""
    my_agent = ai.agent(tools=[double, concat])

    turn1 = [
        tool_call_msg(
            tc_id="tc-1", name="concat", args='{"a": "hello", "b": " world"}'
        )
    ]
    turn2 = [
        tool_call_msg(tc_id="tc-2", name="double", args='{"x": 3}', id="msg-2")
    ]
    turn3 = [text_msg("Done: hello world, 6", id="msg-3")]
    llm = mock_llm([turn1, turn2, turn3])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("Concat then double")]
    ) as stream:
        await collect_messages(stream)
    assert llm.call_count == 3
