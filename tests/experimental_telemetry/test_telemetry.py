"""Core span API: nesting, errors, adapters, replay, span events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from typing import TYPE_CHECKING, Any

import pytest

import ai

from ..conftest import Recorder

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable


async def test_nesting_and_ids(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("outer") as outer:
        assert ai.experimental_telemetry.current() is outer
        async with ai.experimental_telemetry.span("inner") as inner:
            assert inner.parent_id == outer.id
            assert inner.trace_id == outer.trace_id
    assert ai.experimental_telemetry.current() is None
    assert outer.parent_id is None
    assert [s.name for s in recorder.started] == ["outer", "inner"]
    assert [s.name for s in recorder.ended] == ["inner", "outer"]
    assert all(s.ended_at is not None for s in recorder.ended)


async def test_task_inherits_current_span(recorder: Recorder) -> None:
    async def work() -> None:
        async with ai.experimental_telemetry.span("child"):
            pass

    async with ai.experimental_telemetry.span("parent") as parent:
        await asyncio.create_task(work())

    child = next(s for s in recorder.ended if s.name == "child")
    assert child.parent_id == parent.id
    assert child.trace_id == parent.trace_id


async def test_error_recorded_and_reraised(recorder: Recorder) -> None:
    with pytest.raises(ValueError, match="boom"):
        async with ai.experimental_telemetry.span("failing"):
            raise ValueError("boom")
    (span,) = recorder.ended
    assert isinstance(span.error, ValueError)
    assert span.ended_at is not None


async def test_set_attributes(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("s", a=1) as span:
        span.set(b=2)
    (ended,) = recorder.ended
    assert isinstance(ended.data, ai.experimental_telemetry.CustomSpanData)
    assert ended.data.attributes == {"a": 1, "b": 2}


async def test_set_rejects_framework_spans() -> None:
    async with ai.experimental_telemetry.span(
        ai.experimental_telemetry.LoopTurnSpanData()
    ) as span:
        with pytest.raises(TypeError):
            span.set(a=1)


async def test_data_with_attributes_rejected() -> None:
    with pytest.raises(TypeError):
        # The overloads reject this statically too; the runtime check
        # covers untyped callers.
        data = ai.experimental_telemetry.LoopTurnSpanData()
        async with ai.experimental_telemetry.span(
            data,  # ty: ignore[invalid-argument-type]
            a=1,  # type: ignore[call-overload]
        ):
            pass


async def test_replay_flag(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("s", replay=True):
        pass
    assert recorder.ended[0].replay


async def test_not_set_as_current(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("outer") as outer:
        async with ai.experimental_telemetry.span(
            "overlapping", set_as_current=False
        ) as overlapping:
            # The span exists and parents normally...
            assert overlapping.parent_id == outer.id
            # ...but is not current: work opened meanwhile parents to
            # the outer span instead.
            assert ai.experimental_telemetry.current() is outer
            async with ai.experimental_telemetry.span("child") as child:
                assert child.parent_id == outer.id


async def test_out_of_order_close_raises(recorder: Recorder) -> None:
    cm_a = ai.experimental_telemetry.span("a")
    cm_b = ai.experimental_telemetry.span("b")
    await cm_a.__aenter__()
    await cm_b.__aenter__()
    with pytest.raises(RuntimeError, match="closed out of order"):
        await cm_a.__aexit__(None, None, None)
    # The span was still ended for adapters before raising.
    assert [s.name for s in recorder.ended] == ["a"]
    await cm_b.__aexit__(None, None, None)


async def test_failing_adapter_is_isolated(recorder: Recorder) -> None:
    class Broken:
        def on_span_start(self, span: ai.experimental_telemetry.Span) -> None:
            raise RuntimeError("adapter bug")

    broken = Broken()
    ai.experimental_telemetry.register(broken)
    try:
        async with ai.experimental_telemetry.span("s"):
            pass
    finally:
        ai.experimental_telemetry.unregister(broken)
    # The broken adapter neither killed the span nor the other adapter.
    assert [s.name for s in recorder.ended] == ["s"]


async def test_async_adapter_methods_awaited() -> None:
    ended: list[str] = []

    class AsyncAdapter:
        async def on_span_end(
            self, span: ai.experimental_telemetry.Span
        ) -> None:
            ended.append(span.name)

    adapter = AsyncAdapter()
    ai.experimental_telemetry.register(adapter)
    try:
        async with ai.experimental_telemetry.span("s"):
            pass
    finally:
        ai.experimental_telemetry.unregister(adapter)
    assert ended == ["s"]


# ── SpanRef + explicit parent ─────────────────────────────────────


async def test_span_ref_and_current_ref(recorder: Recorder) -> None:
    assert ai.experimental_telemetry.current_ref() is None
    async with ai.experimental_telemetry.span("s") as sp:
        ref = ai.experimental_telemetry.current_ref()
        assert ref is not None
        assert ref == sp.ref
        assert ref.trace_id == sp.trace_id
        assert ref.span_id == sp.id
        assert ref.sampled
    assert ai.experimental_telemetry.current_ref() is None
    # A ref round-trips like any pydantic model.
    restored = ai.experimental_telemetry.SpanRef.model_validate(
        ref.model_dump()
    )
    assert restored == ref


async def test_explicit_span_parent(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("elsewhere") as elsewhere:
        pass
    async with ai.experimental_telemetry.span("ambient"):
        async with ai.experimental_telemetry.span(
            "child", parent=elsewhere
        ) as child:
            # The explicit parent wins over the ambient one...
            assert child.parent_id == elsewhere.id
            assert child.trace_id == elsewhere.trace_id
            # ...and only changes where the span hangs: it is still
            # current inside the block.
            async with ai.experimental_telemetry.span(
                "grandchild"
            ) as grandchild:
                assert grandchild.parent_id == child.id


async def test_span_ref_parent_continues_trace(recorder: Recorder) -> None:
    # "Process one": capture the position as plain data.
    async with ai.experimental_telemetry.span("origin") as origin:
        payload = origin.ref.model_dump()

    # "Process two": restore and continue the same trace.
    ref = ai.experimental_telemetry.SpanRef.model_validate(payload)
    async with ai.experimental_telemetry.span("pickup", parent=ref) as pickup:
        assert pickup.trace_id == origin.trace_id
        assert pickup.parent_id == origin.id
        async with ai.experimental_telemetry.span("nested") as nested:
            assert nested.trace_id == origin.trace_id
            assert nested.parent_id == pickup.id


# ── span/trace ids ────────────────────────────────────────────────


async def test_ids_deterministic_under_use_random(recorder: Recorder) -> None:
    async def run() -> tuple[str, str, str]:
        with ai.messages.use_random(random.Random(7)):
            async with ai.experimental_telemetry.span("outer") as outer:
                async with ai.experimental_telemetry.span("inner") as inner:
                    pass
        return outer.trace_id, outer.id, inner.id

    # The replay contract: re-running the same work under the same
    # random source re-emits spans with identical identities.
    assert await run() == await run()


# ── Adapter base class ────────────────────────────────────────────


async def test_adapter_wrap_span_method_driven() -> None:
    class Vendor(ai.experimental_telemetry.Adapter):
        def __init__(self) -> None:
            self.log: list[str] = []  # instance state, no __init__ chaining

        async def wrap_span(
            self, span: ai.experimental_telemetry.Span
        ) -> AsyncGenerator[None, Any]:
            if span.name == "boring":
                return  # opt out before the first yield
            self.log.append(f"start {span.name}")
            while (ev := (yield)) is not None:
                self.log.append(f"event {ev.name}")
            self.log.append(f"end {span.name}")

    vendor = Vendor()
    async with _registered(vendor):
        async with ai.experimental_telemetry.span("boring"):
            pass
        async with ai.experimental_telemetry.span("outer") as outer:
            async with ai.experimental_telemetry.span("inner"):
                pass
            await outer.add_event("milestone")

    # The base class's hooks drove one generator frame per span, with
    # the same semantics as the wrap_span function.
    assert vendor.log == [
        "start outer",
        "start inner",
        "end inner",
        "event milestone",
        "end outer",
    ]


async def test_adapter_defaults_are_noops(recorder: Recorder) -> None:
    # A bare Adapter is valid: no per-span frames.
    async with _registered(ai.experimental_telemetry.Adapter()):
        async with ai.experimental_telemetry.span("s") as sp:
            await sp.add_event("e")
    assert [s.name for s in recorder.ended] == ["s"]


async def test_adapter_hook_override_composes_with_super() -> None:
    log: list[str] = []

    class Both(ai.experimental_telemetry.Adapter):
        async def wrap_span(
            self, span: ai.experimental_telemetry.Span
        ) -> AsyncGenerator[None, Any]:
            while (yield) is not None:
                pass
            log.append(f"frame end {span.name}")

        async def on_span_end(
            self, span: ai.experimental_telemetry.Span, /
        ) -> None:
            log.append(f"hook end {span.name}")
            # Overriding a hook replaces the driver for that phase;
            # super() plugs it back in.
            await super().on_span_end(span)

    async with _registered(Both()):
        async with ai.experimental_telemetry.span("s"):
            pass

    assert log == ["hook end s", "frame end s"]


async def test_adapter_subclass_state_cannot_collide_with_driver() -> None:
    class Clashy(ai.experimental_telemetry.Adapter):
        def __init__(self) -> None:
            self._live = "mine"  # same name the driver mangles away
            self.ended: list[str] = []

        async def wrap_span(
            self, span: ai.experimental_telemetry.Span
        ) -> AsyncGenerator[None, Any]:
            while (yield) is not None:
                pass
            self.ended.append(span.name)

    clashy = Clashy()
    async with _registered(clashy):
        async with ai.experimental_telemetry.span("s"):
            pass
    assert clashy.ended == ["s"]
    assert clashy._live == "mine"


async def test_wrap_span_function_returns_adapter() -> None:
    @ai.experimental_telemetry.wrap_span
    async def vendor(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass

    assert isinstance(vendor, ai.experimental_telemetry.Adapter)
    assert "wrap_span" in repr(vendor)


# ── wrap_span ─────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def _registered(adapter: Any) -> AsyncIterator[None]:
    ai.experimental_telemetry.register(adapter)
    try:
        yield
    finally:
        ai.experimental_telemetry.unregister(adapter)


async def test_wrap_span_locals_across_yield() -> None:
    events: list[str] = []

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        name = span.name  # local state, no bookkeeping dict
        events.append(f"start {name}")
        while (yield) is not None:
            pass
        assert span.ended_at is not None
        assert isinstance(span.data, ai.experimental_telemetry.CustomSpanData)
        events.append(f"end {name} a={span.data.attributes.get('a')}")

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("outer") as outer:
            async with ai.experimental_telemetry.span("inner") as inner:
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

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        async with vendor_span():
            while (yield) is not None:
                pass

    error = ValueError("boom")
    async with _registered(adapter):
        with pytest.raises(ValueError) as excinfo:
            async with ai.experimental_telemetry.span("failing"):
                raise error
        # The app still gets the original exception...
        assert excinfo.value is error
        # ...and the vendor context manager saw the very same object,
        # as if it had wrapped the work itself.
        assert seen == [error]

        async with ai.experimental_telemetry.span("fine"):
            pass
        assert seen == [error, None]


async def test_wrap_span_swallowing_error_only_local() -> None:
    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        with contextlib.suppress(ValueError):
            while (yield) is not None:
                pass

    async with _registered(adapter):
        # The adapter suppressing the thrown error in its own frame
        # never suppresses it for the application.
        with pytest.raises(ValueError):
            async with ai.experimental_telemetry.span("failing"):
                raise ValueError("boom")


async def test_wrap_span_opt_out_before_yield(recorder: Recorder) -> None:
    ended: list[str] = []

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        if span.name == "boring":
            return
        while (yield) is not None:
            pass
        ended.append(span.name)

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("boring"):
            pass
        async with ai.experimental_telemetry.span("interesting"):
            pass

    assert ended == ["interesting"]
    assert [s.name for s in recorder.ended] == ["boring", "interesting"]


async def test_wrap_span_failures_isolated(recorder: Recorder) -> None:
    @ai.experimental_telemetry.wrap_span
    async def broken_before(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        raise RuntimeError("pre-yield bug")
        while (yield) is not None:
            pass

    @ai.experimental_telemetry.wrap_span
    async def broken_after(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass
        raise RuntimeError("post-loop bug")

    @ai.experimental_telemetry.wrap_span
    async def yields_after_end(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass
        yield  # one yield too many: span already ended

    async with _registered(broken_before):
        async with _registered(broken_after):
            async with _registered(yields_after_end):
                async with ai.experimental_telemetry.span("s"):
                    pass

    # None of the broken generators killed the span or other adapters.
    assert [s.name for s in recorder.ended] == ["s"]


async def test_wrap_span_rejects_plain_functions() -> None:
    async def not_a_generator(span: ai.experimental_telemetry.Span) -> None:
        pass

    fn: Any = not_a_generator
    with pytest.raises(TypeError, match="async generator function"):
        ai.experimental_telemetry.wrap_span(fn)


async def test_wrap_span_events_live_loop() -> None:
    seen: list[str] = []

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        vendor = f"vendor:{span.name}"  # frame-local state
        while (ev := (yield)) is not None:
            # Delivered live, while the span is still open.
            assert span.ended_at is None
            seen.append(f"{vendor} event {ev.name}")
        assert isinstance(span.data, ai.experimental_telemetry.CustomSpanData)
        seen.append(f"{vendor} end a={span.data.attributes.get('a')}")

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("s") as sp:
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

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        try:
            while (ev := (yield)) is not None:
                seen.append(ev.name)
        except ValueError as exc:
            seen.append(f"error {exc}")
            raise

    async with _registered(adapter):
        with pytest.raises(ValueError, match="boom"):
            async with ai.experimental_telemetry.span("failing") as sp:
                await sp.add_event("one")
                raise ValueError("boom")

    # The span's error is thrown in at the yield inside the loop.
    assert seen == ["one", "error boom"]


async def test_wrap_span_early_finish_opts_out(
    recorder: Recorder,
) -> None:
    seen: list[str] = []
    ended: list[str] = []

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        ev = yield  # handle one event, then finish
        assert ev is not None
        seen.append(ev.name)
        ended.append(span.name)

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("s") as sp:
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
    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            raise RuntimeError("event bug")

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("s") as sp:
            await sp.add_event("one")
            await sp.add_event("two")  # generator already dead: skipped

    assert [s.name for s in recorder.ended] == ["s"]


async def test_wrap_span_drain_loop() -> None:
    order: list[str] = []

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        order.append("start")
        while (yield) is not None:
            pass  # a bridge that doesn't react to events drains them
        # The events are still on the span at end, with timestamps.
        order.append(f"end events={[e.name for e in span.span_events]}")

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("s") as sp:
            await sp.add_event("one")
            order.append("event added")

    assert order == ["start", "event added", "end events=['one']"]


# ── span events ───────────────────────────────────────────────────


async def test_add_event_appends_stamps_and_returns(
    recorder: Recorder,
) -> None:
    async with ai.experimental_telemetry.span("s") as sp:
        first = await sp.add_event("first", a=1)
        second = await sp.add_event("second")

    assert sp.span_events == [first, second]
    assert isinstance(first, ai.experimental_telemetry.SpanEvent)
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
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            seen.append(("sync", event.name, span.ended_at is None))

    class AsyncAdapter:
        async def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            seen.append(("async", event.name, span.ended_at is None))

    async with _registered(SyncAdapter()), _registered(AsyncAdapter()):
        async with ai.experimental_telemetry.span("s") as sp:
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
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            raise RuntimeError("adapter bug")

    class Fine:
        def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            seen.append(event.name)

    # Broken registers first: it must not stop dispatch to Fine.
    async with _registered(Broken()), _registered(Fine()):
        async with ai.experimental_telemetry.span("s") as sp:
            event = await sp.add_event("milestone")

    assert seen == ["milestone"]
    assert sp.span_events == [event]
    assert [s.name for s in recorder.ended] == ["s"]


async def test_add_event_after_end_warns_and_appends(
    recorder: Recorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with ai.experimental_telemetry.span("s") as sp:
        pass
    assert sp.ended_at is not None

    with caplog.at_level(
        logging.WARNING, logger="ai.experimental_telemetry.span"
    ):
        late = await sp.add_event("late")

    assert sp.span_events == [late]
    assert any(
        "already-ended" in record.getMessage() for record in caplog.records
    )


# ── inert spans (no adapters registered) ──────────────────────────


async def test_span_without_adapters_is_inert() -> None:
    async with ai.experimental_telemetry.span("outer", a=1) as sp:
        assert sp.id == ""
        assert sp.trace_id == ""
        assert sp.parent_id is None
        assert sp.started_at == 0
        # Never current: nothing to parent under, nothing to ref.
        assert ai.experimental_telemetry.current() is None
        assert ai.experimental_telemetry.current_ref() is None
        # Attribute writes still work; they are just never observed.
        sp.set(b=2)
        event = await sp.add_event("milestone", c=3)
    # add_event returned an inert event and retained nothing.
    assert event.name == "milestone"
    assert event.time_ns == 0
    assert event.attributes == {"c": 3}
    assert sp.span_events == []
    assert sp.ended_at is None


async def test_inert_span_reraises_without_recording() -> None:
    with pytest.raises(ValueError, match="boom"):
        async with ai.experimental_telemetry.span("s") as sp:
            raise ValueError("boom")
    assert sp.error is None


async def test_no_clock_reads_without_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> int:
        raise AssertionError("clock read while telemetry is off")

    monkeypatch.setattr(time, "time_ns", forbidden)
    # The durable-execution contract: unused telemetry makes no
    # non-deterministic calls, so no use_clock setup is needed.
    async with ai.experimental_telemetry.span("outer") as sp:
        await sp.add_event("milestone")
        async with ai.experimental_telemetry.span(
            ai.experimental_telemetry.LoopTurnSpanData()
        ):
            pass


async def test_no_rng_draws_without_adapters() -> None:
    async def id_after_spans(spans: int) -> str:
        with ai.messages.use_random(random.Random(7)):
            for _ in range(spans):
                async with ai.experimental_telemetry.span("s"):
                    pass
            return ai.messages.generate_id()

    # Inert spans consume nothing from the ambient random source, so
    # ids drawn afterwards are unaffected by how many spans opened.
    assert await id_after_spans(0) == await id_after_spans(5)


async def test_register_mid_span_keeps_open_span_inert() -> None:
    r = Recorder()
    async with ai.experimental_telemetry.span("before") as before:
        ai.experimental_telemetry.register(r)
        try:
            # The inert decision is snapshotted at span open.
            assert before.id == ""
            await before.add_event("late")
            async with ai.experimental_telemetry.span("after") as after:
                # Live, but starts its own trace: the inert span
                # cannot parent it.
                assert after.id
                assert after.parent_id is None
        finally:
            ai.experimental_telemetry.unregister(r)

    assert [s.name for s in r.started] == ["after"]
    assert [s.name for s in r.ended] == ["after"]


# ── use_clock ─────────────────────────────────────────────────────


def _ticking_clock(now_ns: int, tick_ns: int = 10) -> Callable[[], int]:
    """Deterministic clock: each reading ticks forward by ``tick_ns``."""

    def time_ns() -> int:
        nonlocal now_ns
        now_ns += tick_ns
        return now_ns

    return time_ns


async def _stamps() -> tuple[int, int, int | None]:
    async with ai.experimental_telemetry.span("s") as sp:
        event = await sp.add_event("milestone")
    return sp.started_at, event.time_ns, sp.ended_at


async def test_use_clock_overrides_and_restores(recorder: Recorder) -> None:
    # A ticking clock gives a deterministic timestamp sequence; the
    # override drives started_at, event stamps, and ended_at alike.
    with ai.experimental_telemetry.use_clock(_ticking_clock(1_000)):
        first = await _stamps()
    with ai.experimental_telemetry.use_clock(_ticking_clock(1_000)):
        second = await _stamps()

    assert first == second == (1_010, 1_020, 1_030)

    # Restored on exit -- back to the wall clock.
    async with ai.experimental_telemetry.span("s") as sp:
        pass
    assert sp.started_at > 1_030


async def test_use_clock_decorator_handles_async_functions(
    recorder: Recorder,
) -> None:
    # Works as a decorator on an async fn; the clock is shared across
    # calls, so the second call continues where the first left off.
    @ai.experimental_telemetry.use_clock(_ticking_clock(1_000))
    async def run() -> tuple[int, int, int | None]:
        return await _stamps()

    assert await run() == (1_010, 1_020, 1_030)
    assert await run() == (1_040, 1_050, 1_060)


async def test_use_clock_accepts_time_time_ns(recorder: Recorder) -> None:
    before = time.time_ns()
    with ai.experimental_telemetry.use_clock(time.time_ns):
        async with ai.experimental_telemetry.span("s") as sp:
            pass
    assert sp.ended_at is not None
    assert before <= sp.started_at <= sp.ended_at <= time.time_ns()
