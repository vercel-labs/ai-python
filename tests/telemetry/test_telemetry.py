"""Telemetry: event sequence, run_id, enable/disable, error capture."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest

import ai
from ai.telemetry.events import (
    RunFinishEvent,
    RunStartEvent,
    TelemetryEvent,
    ToolCallFinishEvent,
    ToolCallStartEvent,
)

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg

# ── Recording handler ────────────────────────────────────────────


class RecordingHandler:
    """Captures all telemetry events in order for assertions."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def handle(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def of_type(self, cls: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, cls)]


@pytest.fixture
def handler() -> Generator[RecordingHandler]:
    h = RecordingHandler()
    ai.telemetry.enable(h)
    yield h
    ai.telemetry.disable()


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    return x * 2


# ── Event sequence: text-only run ────────────────────────────────


async def test_text_only_run_events(handler: RecordingHandler) -> None:
    """Simplest run emits RunStart, StepStart, StepFinish, RunFinish."""
    my_agent = ai.agent()

    mock_llm([[text_msg("Hello!")]])
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
        pass

    types = [type(e).__name__ for e in handler.events]
    assert types == [
        "RunStartEvent",
        "StepStartEvent",
        "StepFinishEvent",
        "RunFinishEvent",
    ]
    assert handler.of_type(RunFinishEvent)[0].error is None


# ── Event sequence: tool call run ────────────────────────────────


async def test_tool_call_events(handler: RecordingHandler) -> None:
    """Tool-calling run emits tool events between steps."""
    my_agent = ai.agent(tools=[double])

    mock_llm(
        [
            [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}')],
            [text_msg("10")],
        ]
    )
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Double 5")]):
        pass

    types = [type(e).__name__ for e in handler.events]
    assert types == [
        "RunStartEvent",
        "StepStartEvent",
        "StepFinishEvent",
        "ToolCallStartEvent",
        "ToolCallFinishEvent",
        "StepStartEvent",
        "StepFinishEvent",
        "RunFinishEvent",
    ]

    tc_start = handler.of_type(ToolCallStartEvent)[0]
    tc_finish = handler.of_type(ToolCallFinishEvent)[0]
    assert tc_start.tool_name == "double"
    assert tc_start.args == '{"x": 5}'
    assert tc_finish.result == 10
    assert tc_finish.error is None
    assert tc_finish.duration_ms >= 0


# ── run_id contextvar ────────────────────────────────────────────


async def test_run_id_available_during_run() -> None:
    """get_run_id() returns a non-empty ID inside a handler during run."""
    captured: str = ""

    class Capture:
        def handle(self, event: TelemetryEvent) -> None:
            nonlocal captured
            if isinstance(event, RunStartEvent):
                captured = ai.telemetry.get_run_id()

    ai.telemetry.enable(Capture())
    try:
        my_agent = ai.agent()

        mock_llm([[text_msg("Hello!")]])
        async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
            pass
        assert len(captured) == 16
    finally:
        ai.telemetry.disable()


# ── enable / disable lifecycle ───────────────────────────────────


async def test_disable_reverts_to_noop() -> None:
    """disable() stops events from reaching the handler."""
    handler = RecordingHandler()
    ai.telemetry.enable(handler)

    my_agent = ai.agent()

    mock_llm([[text_msg("Hello!")]])
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
        pass
    assert len(handler.of_type(RunStartEvent)) == 1

    ai.telemetry.disable()
    handler.events.clear()

    mock_llm([[text_msg("Hello!")]])
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
        pass
    assert len(handler.events) == 0


# ── User-emitted custom events ──────────────────────────────────


async def test_user_emitted_custom_event(handler: RecordingHandler) -> None:
    """ai.telemetry.handle() delivers user events to the active handler."""

    @dataclasses.dataclass(frozen=True, slots=True)
    class CustomEvent(TelemetryEvent):
        message: str

    my_agent = ai.agent()

    @my_agent.loop
    async def custom(context: ai.Context) -> AsyncGenerator[ai.Message]:
        ai.telemetry.handle(CustomEvent(message="hello"))
        async for msg in await ai.models.stream(context.model, context.messages):
            yield msg

    mock_llm([[text_msg("Hello!")]])
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
        pass

    custom_events = [e for e in handler.events if isinstance(e, CustomEvent)]
    assert len(custom_events) == 1
    assert custom_events[0].message == "hello"


# ── Error capture ────────────────────────────────────────────────


async def test_run_error_in_finish_event(handler: RecordingHandler) -> None:
    """RunFinishEvent captures the error when the loop function raises."""
    my_agent = ai.agent()

    @my_agent.loop
    async def failing(context: ai.Context) -> AsyncGenerator[ai.Message]:
        raise ValueError("boom")
        yield  # make it a generator

    mock_llm([])
    with pytest.raises(ExceptionGroup):
        async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
            pass

    run_finish = handler.of_type(RunFinishEvent)[0]
    assert run_finish.error is not None
