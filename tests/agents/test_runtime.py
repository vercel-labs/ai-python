"""Agent default loop, tool execution, multi-turn."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

import ai
from ai.agents import runtime
from ai.types import events, messages

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


# -- runtime.run ------------------------------------------------------------


async def test_run_source_lockstep() -> None:
    """run() only advances its source when the consumer asks for an event."""
    advanced: list[int] = []

    async def src() -> AsyncGenerator[events.AgentEvent]:
        for i in range(10):
            advanced.append(i)
            yield events.TextDelta(chunk=str(i))

    it = runtime.run(src())
    async with contextlib.aclosing(it):
        for n in range(5):
            ev = await anext(it)
            assert isinstance(ev, events.TextDelta)
            assert ev.chunk == str(n)
            # Give the pipeline every opportunity to run ahead.
            for _ in range(50):
                await asyncio.sleep(0)
            assert advanced == list(range(n + 1))


async def test_run_delivers_put_events() -> None:
    """Events put on the Runtime queue by the source are yielded too."""

    async def src() -> AsyncGenerator[events.AgentEvent]:
        await runtime.get_runtime().put_event(events.TextDelta(chunk="side"))
        yield events.TextDelta(chunk="main")

    chunks: set[str] = set()
    async with contextlib.aclosing(runtime.run(src())) as it:
        async for ev in it:
            assert isinstance(ev, events.TextDelta)
            chunks.add(ev.chunk)
    assert chunks == {"side", "main"}


# -- Agent.run buffer -------------------------------------------------------


class _CountingAgent(ai.Agent):
    """Loop that records how far it has advanced."""

    def __init__(self) -> None:
        super().__init__()
        self.advanced: list[int] = []

    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[events.AgentEvent]:
        for i in range(10):
            self.advanced.append(i)
            yield events.TextDelta(chunk=str(i))


async def test_agent_run_buffers_by_default() -> None:
    """With the default buffer, the loop keeps going while the consumer
    is between reads."""
    agent = _CountingAgent()
    async with agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        await anext(aiter(stream))
        for _ in range(50):
            await asyncio.sleep(0)
        assert agent.advanced == list(range(10))
        assert [
            e.chunk async for e in stream if isinstance(e, events.TextDelta)
        ] == [str(i) for i in range(1, 10)]


class _LockstepAgent(_CountingAgent):
    LOOP_BUFFER = 0


async def test_agent_run_loop_buffer_zero_is_lockstep() -> None:
    """With LOOP_BUFFER = 0 the loop is only advanced when the consumer
    asks."""
    agent = _LockstepAgent()
    async with agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        it = aiter(stream)
        for n in range(5):
            ev = await anext(it)
            assert isinstance(ev, events.TextDelta)
            assert ev.chunk == str(n)
            for _ in range(50):
                await asyncio.sleep(0)
            assert agent.advanced == list(range(n + 1))


# -- Agent default loop: single turn (no tools) ----------------------------


async def test_agent_text_only() -> None:
    """Agent default loop with no tool calls returns after one LLM call."""
    my_agent = ai.Agent(tools=[double])

    llm = mock_llm([[text_msg("Hello!")]])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]) as stream:
        msgs = await collect_messages(stream)
    assert llm.call_count == 1
    assert any(m.text == "Hello!" for m in msgs)


# -- Agent default loop: tool call + follow-up -----------------------------


async def test_agent_tool_then_text() -> None:
    """Agent default loop calls tool, feeds result back, gets final text."""
    my_agent = ai.Agent(tools=[double])

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
    tr = tool_results[0].tool_results[0].result
    assert tr == 10


# -- Agent default loop: multiple tool calls in one message ----------------


async def test_agent_parallel_tools() -> None:
    """LLM returns two tool calls in one message; both execute."""
    my_agent = ai.Agent(tools=[double])

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
    my_agent = ai.Agent(tools=[double, concat])

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
