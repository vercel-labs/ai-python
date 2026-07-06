"""Core span API: nesting, errors, adapters, replay."""

from __future__ import annotations

import asyncio

import pytest

import ai

from ..conftest import Recorder


async def test_nesting_and_ids(recorder: Recorder) -> None:
    async with ai.telemetry.span("outer") as outer:
        assert ai.telemetry.current() is outer
        async with ai.telemetry.span("inner") as inner:
            assert inner.parent_id == outer.id
            assert inner.trace_id == outer.trace_id
    assert ai.telemetry.current() is None
    assert outer.parent_id is None
    assert [s.name for s in recorder.started] == ["outer", "inner"]
    assert [s.name for s in recorder.ended] == ["inner", "outer"]
    assert all(s.ended_at is not None for s in recorder.ended)


async def test_task_inherits_current_span(recorder: Recorder) -> None:
    async def work() -> None:
        async with ai.telemetry.span("child"):
            pass

    async with ai.telemetry.span("parent") as parent:
        await asyncio.create_task(work())

    child = next(s for s in recorder.ended if s.name == "child")
    assert child.parent_id == parent.id
    assert child.trace_id == parent.trace_id


async def test_error_recorded_and_reraised(recorder: Recorder) -> None:
    with pytest.raises(ValueError, match="boom"):
        async with ai.telemetry.span("failing"):
            raise ValueError("boom")
    (span,) = recorder.ended
    assert isinstance(span.error, ValueError)
    assert span.ended_at is not None


async def test_set_attributes(recorder: Recorder) -> None:
    async with ai.telemetry.span("s", a=1) as span:
        span.set(b=2)
    (ended,) = recorder.ended
    assert isinstance(ended.data, ai.telemetry.CustomSpanData)
    assert ended.data.attributes == {"a": 1, "b": 2}


async def test_set_rejects_framework_spans() -> None:
    async with ai.telemetry.span(ai.telemetry.StepSpanData(index=1)) as span:
        with pytest.raises(TypeError):
            span.set(a=1)


async def test_data_with_attributes_rejected() -> None:
    with pytest.raises(TypeError):
        async with ai.telemetry.span(ai.telemetry.StepSpanData(index=1), a=1):
            pass


async def test_replay_flag(recorder: Recorder) -> None:
    async with ai.telemetry.span("s", replay=True):
        pass
    assert recorder.ended[0].replay


async def test_not_set_as_current(recorder: Recorder) -> None:
    async with ai.telemetry.span("outer") as outer:
        async with ai.telemetry.span(
            "overlapping", set_as_current=False
        ) as overlapping:
            # The span exists and parents normally...
            assert overlapping.parent_id == outer.id
            # ...but is not current: work opened meanwhile parents to
            # the outer span instead.
            assert ai.telemetry.current() is outer
            async with ai.telemetry.span("child") as child:
                assert child.parent_id == outer.id


async def test_out_of_order_close_raises(recorder: Recorder) -> None:
    cm_a = ai.telemetry.span("a")
    cm_b = ai.telemetry.span("b")
    await cm_a.__aenter__()
    await cm_b.__aenter__()
    with pytest.raises(RuntimeError, match="closed out of order"):
        await cm_a.__aexit__(None, None, None)
    # The span was still ended for adapters before raising.
    assert [s.name for s in recorder.ended] == ["a"]
    await cm_b.__aexit__(None, None, None)


async def test_failing_adapter_is_isolated(recorder: Recorder) -> None:
    class Broken:
        def on_span_start(self, span: ai.telemetry.Span) -> None:
            raise RuntimeError("adapter bug")

    broken = Broken()
    ai.telemetry.register(broken)
    try:
        async with ai.telemetry.span("s"):
            pass
    finally:
        ai.telemetry.unregister(broken)
    # The broken adapter neither killed the span nor the other adapter.
    assert [s.name for s in recorder.ended] == ["s"]


async def test_async_adapter_methods_awaited() -> None:
    ended: list[str] = []

    class AsyncAdapter:
        async def on_span_end(self, span: ai.telemetry.Span) -> None:
            ended.append(span.name)

    adapter = AsyncAdapter()
    ai.telemetry.register(adapter)
    try:
        async with ai.telemetry.span("s"):
            pass
    finally:
        ai.telemetry.unregister(adapter)
    assert ended == ["s"]
