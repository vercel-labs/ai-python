"""Telemetry spans emitted by ``Agent.run``: the run/turn/tool tree."""

from __future__ import annotations

import ai

from ..conftest import MOCK_MODEL, Recorder, mock_llm, text_msg, tool_call_msg


def _by_name(recorder: Recorder) -> dict[str, list[ai.telemetry.Span]]:
    spans: dict[str, list[ai.telemetry.Span]] = {}
    for s in recorder.ended:
        spans.setdefault(s.name, []).append(s)
    return spans


async def test_agent_run_span_tree(recorder: Recorder) -> None:
    @ai.tool
    async def lookup(x: int) -> str:
        """Tool that opens a user span."""
        async with ai.telemetry.span("user_work", x=x):
            return "ok"

    mock_llm(
        [
            [tool_call_msg(name="lookup", args='{"x": 1}')],
            [text_msg("done", id="msg-2")],
        ]
    )
    my_agent = ai.agent(tools=[lookup])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for _ in stream:
            pass

    spans = _by_name(recorder)
    (run,) = spans["run"]
    steps = sorted(spans["loop_turn"], key=lambda s: s.started_at)
    calls = sorted(spans["ai_stream"], key=lambda s: s.started_at)
    (tool_span,) = spans["tool_execution"]
    (user_span,) = spans["user_work"]

    # Shape: run > turn[1,2]; turn 1 > ai_stream + tool_execution > user span.
    assert len(steps) == 2
    assert len(calls) == 2
    assert {s.parent_id for s in steps} == {run.id}
    assert calls[0].parent_id == steps[0].id
    assert calls[1].parent_id == steps[1].id
    assert tool_span.parent_id == steps[0].id
    assert user_span.parent_id == tool_span.id
    assert {s.trace_id for s in recorder.ended} == {run.trace_id}
    assert not any(s.replay for s in recorder.ended)

    assert isinstance(run.data, ai.telemetry.RunSpanData)
    assert run.data.agent == "Agent"
    assert run.data.model == "mock-model"

    assert isinstance(tool_span.data, ai.telemetry.ToolExecutionSpanData)
    assert tool_span.data.tool_name == "lookup"
    assert tool_span.data.args == {"x": 1}
    assert tool_span.data.result == "ok"
    assert not tool_span.data.is_error

    assert isinstance(calls[1].data, ai.telemetry.AiStreamSpanData)
    assert calls[1].data.message is not None
    assert calls[1].data.message.text == "done"


async def test_tool_error_marked_on_span(recorder: Recorder) -> None:
    @ai.tool
    async def boom() -> str:
        """Tool that always fails."""
        raise ValueError("nope")

    mock_llm([[tool_call_msg(name="boom")], [text_msg("done")]])
    my_agent = ai.agent(tools=[boom])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for _ in stream:
            pass

    (tool_span,) = _by_name(recorder)["tool_execution"]
    assert isinstance(tool_span.data, ai.telemetry.ToolExecutionSpanData)
    # The framework converts the exception into an error result, so the
    # span itself succeeded but carries the error marker.
    assert tool_span.error is None
    assert tool_span.data.is_error
