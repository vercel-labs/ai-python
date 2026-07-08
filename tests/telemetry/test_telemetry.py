"""Core span API: nesting, errors, adapters, replay, span events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

import pytest

import ai

from ..conftest import Recorder

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator


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
    async with ai.telemetry.span(ai.telemetry.LoopTurnSpanData()) as span:
        with pytest.raises(TypeError):
            span.set(a=1)


async def test_data_with_attributes_rejected() -> None:
    with pytest.raises(TypeError):
        async with ai.telemetry.span(ai.telemetry.LoopTurnSpanData(), a=1):
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


# ── wrap_span ─────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def _registered(adapter: Any) -> AsyncIterator[None]:
    ai.telemetry.register(adapter)
    try:
        yield
    finally:
        ai.telemetry.unregister(adapter)


async def test_wrap_span_locals_across_yield() -> None:
    events: list[str] = []

    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        name = span.name  # local state, no bookkeeping dict
        events.append(f"start {name}")
        while (yield) is not None:
            pass
        assert span.ended_at is not None
        assert isinstance(span.data, ai.telemetry.CustomSpanData)
        events.append(f"end {name} a={span.data.attributes.get('a')}")

    async with _registered(adapter):
        async with ai.telemetry.span("outer") as outer:
            async with ai.telemetry.span("inner") as inner:
                inner.set(a=1)
            outer.set(a=2)

    # One generator frame per live span, each resumed at its own
    # span's end with the final data visible after the yield.
    assert events == [
        "start outer",
        "start inner",
        "end inner a=1",
        "end outer a=2",
    ]


async def test_wrap_span_vendor_context_manager_sees_error() -> None:
    seen: list[BaseException | None] = []

    @contextlib.asynccontextmanager
    async def vendor_span() -> AsyncIterator[None]:
        try:
            yield
        except BaseException as exc:
            seen.append(exc)
            raise
        else:
            seen.append(None)

    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        async with vendor_span():
            while (yield) is not None:
                pass

    error = ValueError("boom")
    async with _registered(adapter):
        with pytest.raises(ValueError) as excinfo:
            async with ai.telemetry.span("failing"):
                raise error
        # The app still gets the original exception...
        assert excinfo.value is error
        # ...and the vendor context manager saw the very same object,
        # as if it had wrapped the work itself.
        assert seen == [error]

        async with ai.telemetry.span("fine"):
            pass
        assert seen == [error, None]


async def test_wrap_span_swallowing_error_only_local() -> None:
    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        with contextlib.suppress(ValueError):
            while (yield) is not None:
                pass

    async with _registered(adapter):
        # The adapter suppressing the thrown error in its own frame
        # never suppresses it for the application.
        with pytest.raises(ValueError):
            async with ai.telemetry.span("failing"):
                raise ValueError("boom")


async def test_wrap_span_opt_out_before_yield(recorder: Recorder) -> None:
    ended: list[str] = []

    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        if span.name == "boring":
            return
        while (yield) is not None:
            pass
        ended.append(span.name)

    async with _registered(adapter):
        async with ai.telemetry.span("boring"):
            pass
        async with ai.telemetry.span("interesting"):
            pass

    assert ended == ["interesting"]
    assert [s.name for s in recorder.ended] == ["boring", "interesting"]


async def test_wrap_span_failures_isolated(recorder: Recorder) -> None:
    @ai.telemetry.wrap_span
    async def broken_before(
        span: ai.telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        raise RuntimeError("pre-yield bug")
        while (yield) is not None:
            pass

    @ai.telemetry.wrap_span
    async def broken_after(
        span: ai.telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass
        raise RuntimeError("post-loop bug")

    @ai.telemetry.wrap_span
    async def yields_after_end(
        span: ai.telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass
        yield  # one yield too many: span already ended

    async with _registered(broken_before):
        async with _registered(broken_after):
            async with _registered(yields_after_end):
                async with ai.telemetry.span("s"):
                    pass

    # None of the broken generators killed the span or other adapters.
    assert [s.name for s in recorder.ended] == ["s"]


async def test_wrap_span_rejects_plain_functions() -> None:
    async def not_a_generator(span: ai.telemetry.Span) -> None:
        pass

    fn: Any = not_a_generator
    with pytest.raises(TypeError, match="async generator function"):
        ai.telemetry.wrap_span(fn)


async def test_wrap_span_events_live_loop() -> None:
    seen: list[str] = []

    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        vendor = f"vendor:{span.name}"  # frame-local state
        while (ev := (yield)) is not None:
            # Delivered live, while the span is still open.
            assert span.ended_at is None
            seen.append(f"{vendor} event {ev.name}")
        assert isinstance(span.data, ai.telemetry.CustomSpanData)
        seen.append(f"{vendor} end a={span.data.attributes.get('a')}")

    async with _registered(adapter):
        async with ai.telemetry.span("s") as sp:
            await sp.add_event("one")
            await sp.add_event("two")
            sp.set(a=1)

    assert seen == [
        "vendor:s event one",
        "vendor:s event two",
        "vendor:s end a=1",
    ]


async def test_wrap_span_error_reaches_loop() -> None:
    seen: list[str] = []

    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        try:
            while (ev := (yield)) is not None:
                seen.append(ev.name)
        except ValueError as exc:
            seen.append(f"error {exc}")
            raise

    async with _registered(adapter):
        with pytest.raises(ValueError, match="boom"):
            async with ai.telemetry.span("failing") as sp:
                await sp.add_event("one")
                raise ValueError("boom")

    # The span's error is thrown in at the yield inside the loop.
    assert seen == ["one", "error boom"]


async def test_wrap_span_early_finish_opts_out(
    recorder: Recorder,
) -> None:
    seen: list[str] = []
    ended: list[str] = []

    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        ev = yield  # handle one event, then finish
        assert ev is not None
        seen.append(ev.name)
        ended.append(span.name)

    async with _registered(adapter):
        async with ai.telemetry.span("s") as sp:
            await sp.add_event("one")
            await sp.add_event("two")

    # Finishing mid-span opted out of the rest: the second event and
    # the span end were skipped, and nothing blew up.
    assert seen == ["one"]
    assert ended == ["s"]
    assert [s.name for s in recorder.ended] == ["s"]


async def test_wrap_span_raising_event_handler_isolated(
    recorder: Recorder,
) -> None:
    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            raise RuntimeError("event bug")

    async with _registered(adapter):
        async with ai.telemetry.span("s") as sp:
            await sp.add_event("one")
            await sp.add_event("two")  # generator already dead: skipped

    assert [s.name for s in recorder.ended] == ["s"]


async def test_wrap_span_drain_loop() -> None:
    order: list[str] = []

    @ai.telemetry.wrap_span
    async def adapter(span: ai.telemetry.Span) -> AsyncGenerator[None, Any]:
        order.append("start")
        while (yield) is not None:
            pass  # a bridge that doesn't react to events drains them
        # The events are still on the span at end, with timestamps.
        order.append(f"end events={[e.name for e in span.span_events]}")

    async with _registered(adapter):
        async with ai.telemetry.span("s") as sp:
            await sp.add_event("one")
            order.append("event added")

    assert order == ["start", "event added", "end events=['one']"]


# ── span events ───────────────────────────────────────────────────


async def test_add_event_appends_stamps_and_returns() -> None:
    async with ai.telemetry.span("s") as sp:
        first = await sp.add_event("first", a=1)
        second = await sp.add_event("second")

    assert sp.span_events == [first, second]
    assert isinstance(first, ai.telemetry.SpanEvent)
    assert first.name == "first"
    assert first.attributes == {"a": 1}
    assert second.attributes == {}
    # Wall-clock stamps, appended in call order.
    assert sp.started_at <= first.time_ns <= second.time_ns
    assert sp.ended_at is not None
    assert second.time_ns <= sp.ended_at


async def test_span_event_dispatched_live_sync_and_async(
    recorder: Recorder,
) -> None:
    seen: list[tuple[str, str, bool]] = []

    class SyncAdapter:
        def on_span_event(
            self, span: ai.telemetry.Span, event: ai.telemetry.SpanEvent
        ) -> None:
            seen.append(("sync", event.name, span.ended_at is None))

    class AsyncAdapter:
        async def on_span_event(
            self, span: ai.telemetry.Span, event: ai.telemetry.SpanEvent
        ) -> None:
            seen.append(("async", event.name, span.ended_at is None))

    async with _registered(SyncAdapter()), _registered(AsyncAdapter()):
        async with ai.telemetry.span("s") as sp:
            await sp.add_event("milestone")

    # Both handlers saw the event while the span was still live; the
    # recorder (no on_span_event) was skipped without error.
    assert seen == [
        ("sync", "milestone", True),
        ("async", "milestone", True),
    ]
    assert [s.name for s in recorder.ended] == ["s"]


async def test_span_event_raising_handler_isolated(
    recorder: Recorder,
) -> None:
    seen: list[str] = []

    class Broken:
        def on_span_event(
            self, span: ai.telemetry.Span, event: ai.telemetry.SpanEvent
        ) -> None:
            raise RuntimeError("adapter bug")

    class Fine:
        def on_span_event(
            self, span: ai.telemetry.Span, event: ai.telemetry.SpanEvent
        ) -> None:
            seen.append(event.name)

    # Broken registers first: it must not stop dispatch to Fine.
    async with _registered(Broken()), _registered(Fine()):
        async with ai.telemetry.span("s") as sp:
            event = await sp.add_event("milestone")

    assert seen == ["milestone"]
    assert sp.span_events == [event]
    assert [s.name for s in recorder.ended] == ["s"]


async def test_add_event_after_end_warns_and_appends(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with ai.telemetry.span("s") as sp:
        pass
    assert sp.ended_at is not None

    with caplog.at_level(logging.WARNING, logger="ai.telemetry.span"):
        late = await sp.add_event("late")

    assert sp.span_events == [late]
    assert any(
        "already-ended" in record.getMessage() for record in caplog.records
    )
