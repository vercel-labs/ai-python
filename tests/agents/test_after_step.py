"""Default-loop step boundaries through the public Agent.run API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

import ai
from ai.types import events, messages

from ..conftest import (
    MOCK_MODEL,
    MOCK_PROVIDER,
    Recorder,
    collect_messages,
    mock_llm,
    text_msg,
    tool_call_msg,
)


@pytest.mark.parametrize("buffer", [None, 0])
async def test_after_step_can_continue_final_response(
    buffer: int | None,
) -> None:
    snapshots: list[list[str]] = []
    consumed_responses: list[str] = []

    class PendingInputAgent(ai.Agent):
        LOOP_BUFFER = buffer

        async def after_step(self, context: ai.Context) -> None:
            snapshots.append([message.text for message in context.messages])
            if buffer == 0:
                assert len(consumed_responses) == len(snapshots)
                assert consumed_responses[-1] == context.messages[-1].text
            if len(snapshots) == 1:
                context.add(ai.user_message("One more question"))

    llm = mock_llm([[text_msg("First answer")], [text_msg("Second answer")]])
    async with PendingInputAgent().run(
        MOCK_MODEL, [ai.user_message("Question")]
    ) as stream:
        async for event in stream:
            if isinstance(event, events.StreamEnd):
                consumed_responses.append(event.message.text)
        assert stream.output == "Second answer"

    assert llm.call_count == 2
    assert snapshots == [
        ["Question", "First answer"],
        ["Question", "First answer", "One more question", "Second answer"],
    ]


@pytest.mark.parametrize("first", [1, 2])
@pytest.mark.parametrize("buffer", [None, 0])
async def test_after_step_sees_complete_concurrent_batch(
    first: int, buffer: int | None, recorder: Recorder
) -> None:
    started: list[int] = []
    finished: list[int] = []
    both_started = asyncio.Event()
    first_finished = asyncio.Event()
    snapshots: list[list[messages.Message]] = []
    consumed_results: set[str] = set()

    @ai.tool
    async def complete(number: int) -> int:
        """Complete in the specified order after both tools have started."""
        started.append(number)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        if number != first:
            await first_finished.wait()
        finished.append(number)
        if number == first:
            first_finished.set()
        return number * 10

    class BatchAgent(ai.Agent):
        LOOP_BUFFER = buffer

        async def after_step(self, context: ai.Context) -> None:
            assert finished == [first, 3 - first]
            # The outermost step context exits after stream/runner/merge.
            assert recorder.ended[-1].name == "loop_turn"
            if buffer == 0:
                assert consumed_results == {"tc-1", "tc-2"}
            snapshots.append(list(context.messages))

    calls = messages.Message(
        role="assistant",
        parts=[
            messages.ToolCallPart(
                tool_call_id=f"tc-{number}",
                tool_name="complete",
                tool_args=f'{{"number": {number}}}',
            )
            for number in (1, 2)
        ],
    )
    llm = mock_llm([[calls], [text_msg("Done")]])
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
    assert [message.role for message in snapshots[0]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert len(snapshots[0][1].tool_calls) == 2
    assert {
        result.tool_call_id: result.result
        for result in snapshots[0][-1].tool_results
    } == {"tc-1": 10, "tc-2": 20}
    assert snapshots[1][-1].text == "Done"


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_after_step_failure_stops_before_next_model(
    failure: type[RuntimeError] | type[asyncio.CancelledError],
) -> None:
    calls = 0

    @ai.tool
    async def complete() -> str:
        """Return a result that would ordinarily request another model step."""
        return "result"

    class FailingAgent(ai.Agent):
        async def after_step(self, context: ai.Context) -> None:
            nonlocal calls
            calls += 1
            assert context.messages[-1].tool_results[0].result == "result"
            raise failure("after-step failure")

    llm = mock_llm(
        [
            [tool_call_msg(name="complete")],
            [text_msg("Should not be requested")],
        ]
    )
    expected = ExceptionGroup if failure is RuntimeError else failure
    with pytest.raises(expected) as caught:
        async with FailingAgent(tools=[complete]).run(
            MOCK_MODEL, [ai.user_message("Run tool")]
        ) as stream:
            await collect_messages(stream)

    error = caught.value
    if isinstance(error, ExceptionGroup):
        (error,) = error.exceptions
    assert isinstance(error, failure)
    assert str(error) == "after-step failure"
    assert calls == 1
    assert llm.call_count == 1


async def test_after_step_skips_failed_model_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class ObservingAgent(ai.Agent):
        async def after_step(self, context: ai.Context) -> None:
            nonlocal calls
            calls += 1

    async def failing_stream(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[events.Event]:
        yield events.StreamStart()
        raise RuntimeError("provider disconnected")

    monkeypatch.setattr(MOCK_PROVIDER, "_stream_impl", failing_stream)
    with pytest.raises(ExceptionGroup, match="unhandled errors") as caught:
        async with ObservingAgent().run(
            MOCK_MODEL, [ai.user_message("Question")]
        ) as stream:
            await collect_messages(stream)

    assert "provider disconnected" in str(caught.value.exceptions)
    assert calls == 0
