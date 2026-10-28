from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Sequence

import ai
from ai import durable
from ai.types import events, messages

from .conftest import MOCK_MODEL

log: list[str] = []


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    log.append(f"double:{x}")
    return x * 2


@ai.tool
async def negate(x: int) -> int:
    """Negate a number."""
    log.append(f"negate:{x}")
    return -x


def scripted_streamer(
    steps: list[tuple[Sequence[events.AgentEvent], messages.Message]],
) -> durable.Streamer:
    it = iter(steps)

    @contextlib.asynccontextmanager
    async def streamer(
        *, context: ai.Context
    ) -> AsyncIterator[AsyncGenerator[events.AgentEvent]]:
        early, message = next(it)

        async def stream() -> AsyncGenerator[events.AgentEvent]:
            for event in early:
                yield event
            # stand-in for waiting on the llm step
            for _ in range(10):
                await asyncio.sleep(0)
            log.append("replay")
            async with ai.Stream.replay_message(message) as replay:
                async for event in replay:
                    yield event

        yield stream()

    return streamer


async def test_durable_agent_eager_tools() -> None:
    log.clear()
    a = ai.testing.tool_call(double, x=1)
    b = ai.testing.tool_call(negate, x=2)
    dropped = ai.testing.tool_call(double, x=3)
    streamer = scripted_streamer(
        [
            (
                [
                    events.ToolEnd(tool_call_id=a.tool_call_id, tool_call=a),
                    events.ToolEnd(tool_call_id=b.tool_call_id, tool_call=b),
                    events.ToolEnd(
                        tool_call_id=dropped.tool_call_id, tool_call=dropped
                    ),
                ],
                ai.assistant_message(a, b),
            ),
            ((), ai.assistant_message("done")),
        ]
    )
    agent = durable.DurableAgent(
        tools=[double, negate], streamer=streamer, eager_tools={"double"}
    )

    async with agent.run(MOCK_MODEL, [ai.user_message("go")]) as run:
        async for _ in run:
            pass

    # eager tools start before the final message arrives and aren't rerun
    assert log == ["double:1", "double:3", "replay", "negate:2", "replay"]
    tool_message = run.messages[2]
    assert {r.tool_call_id: r.result for r in tool_message.tool_results} == {
        a.tool_call_id: 2,
        b.tool_call_id: -2,
    }
    assert run.messages[-1].text == "done"
    assert len(run.messages) == 4
