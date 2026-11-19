"""Default-loop step boundaries through the public Agent.run API."""

from __future__ import annotations

import asyncio

import ai
from ai.types import events, messages

from ..conftest import (
    MOCK_MODEL,
    collect_messages,
    mock_llm,
    text_msg,
    tool_call_msg,
)


async def test_after_step_can_continue_final_response() -> None:
    snapshots: list[list[str]] = []

    class PendingInputAgent(ai.Agent):
        async def after_step(self, context: ai.Context) -> None:
            snapshots.append([message.text for message in context.messages])
            if len(snapshots) == 1:
                context.add(ai.user_message("One more question"))

    llm = mock_llm([[text_msg("First answer")], [text_msg("Second answer")]])
    async with PendingInputAgent().run(
        MOCK_MODEL, [ai.user_message("Question")]
    ) as stream:
        await collect_messages(stream)
        assert stream.output == "Second answer"

    assert llm.call_count == 2
    assert snapshots == [
        ["Question", "First answer"],
        ["Question", "First answer", "One more question", "Second answer"],
    ]


async def test_after_step_sees_complete_concurrent_batch() -> None:
    started: list[int] = []
    both_started = asyncio.Event()
    second_finished = asyncio.Event()
    snapshots: list[list[messages.Message]] = []
    consumed_results: set[str] = set()

    @ai.tool
    async def complete(number: int) -> int:
        """Start both tools, then complete the second one first."""
        started.append(number)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        if number == 1:
            await second_finished.wait()
        else:
            second_finished.set()
        return number * 10

    class BatchAgent(ai.Agent):
        LOOP_BUFFER = 0

        async def after_step(self, context: ai.Context) -> None:
            assert consumed_results == {"tc-1", "tc-2"}
            snapshots.append(list(context.messages))

    calls = [
        tool_call_msg(
            tc_id=f"tc-{number}",
            name="complete",
            args=f'{{"number": {number}}}',
        )
        for number in (1, 2)
    ]
    llm = mock_llm([calls, [text_msg("Done")]])
    async with asyncio.timeout(5):
        async with BatchAgent(tools=[complete]).run(
            MOCK_MODEL, [ai.user_message("Run both")]
        ) as stream:
            async for event in stream:
                if isinstance(event, events.ToolCallResult):
                    consumed_results.update(
                        result.tool_call_id for result in event.results
                    )

    assert llm.call_count == 2
    assert len(snapshots) == 2
    assert {
        result.tool_call_id: result.result
        for result in snapshots[0][-1].tool_results
    } == {"tc-1": 10, "tc-2": 20}
    assert snapshots[1][-1].text == "Done"
