"""Core span API: nesting, errors, adapters, replay, span events."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import random
import time
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pydantic
import pytest

import ai

from ..conftest import Recorder

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable


async def _add_event(
    sp: ai.experimental_telemetry.Span,
    name: str,
    **attributes: Any,
) -> ai.experimental_telemetry.SpanEvent:
    """The manual event pattern: stamp, append, push."""
    event = ai.experimental_telemetry.SpanEvent(
        name=name,
        time_ns=ai.experimental_telemetry.now_ns(),
        attributes=attributes,
    )
    sp.events.append(event)
    await sp.push()
    return event


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
    # The error is serializable data, not a live exception.
    assert span.error == ai.experimental_telemetry.SpanError(
        type="ValueError", message="boom"
    )
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


# ── Serializable spans + explicit parent ──────────────────────────


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


async def test_restored_span_parent_continues_trace(
    recorder: Recorder,
) -> None:
    # "Process one": the span itself is the serializable position.
    async with ai.experimental_telemetry.span("origin") as origin:
        payload = origin.model_dump(mode="json")

    # "Process two": restore and continue the same trace.
    restored = ai.experimental_telemetry.Span.model_validate(payload)
    async with ai.experimental_telemetry.span(
        "pickup", parent=restored
    ) as pickup:
        assert pickup.trace_id == origin.trace_id
        assert pickup.parent_id == origin.id
        async with ai.experimental_telemetry.span("nested") as nested:
            assert nested.trace_id == origin.trace_id
            assert nested.parent_id == pickup.id


async def test_use_span_parents_without_lifecycle(
    recorder: Recorder,
) -> None:
    # ``use_span`` is pure context plumbing: no pushes, no timestamps.
    outer = ai.experimental_telemetry.create_span("outer")
    with ai.experimental_telemetry.use_span(outer):
        assert ai.experimental_telemetry.current() is outer
        async with ai.experimental_telemetry.span("child") as child:
            assert child.parent_id == outer.id
            assert child.trace_id == outer.trace_id
    assert ai.experimental_telemetry.current() is None
    # Only the child was reported; outer was never pushed.
    assert [s.name for s in recorder.started] == ["child"]


async def test_span_round_trips_typed_data() -> None:
    data = ai.experimental_telemetry.ToolExecutionSpanData(
        tool_name="lookup", tool_call_id="tc-1", args={"x": 1}
    )
    async with ai.experimental_telemetry.span(data) as sp:
        sp.data.result = "ok"
    payload = sp.model_dump(mode="json")

    # A bare restore rebuilds the framework data type (matched by the
    # ``kind`` tag serialized in the data), so adapters dispatch on
    # re-pushed spans like live ones.
    restored = ai.experimental_telemetry.Span.model_validate(payload)
    assert restored.data == ai.experimental_telemetry.ToolExecutionSpanData(
        tool_name="lookup",
        tool_call_id="tc-1",
        args={"x": 1},
        result="ok",
    )
    assert restored.id == sp.id
    assert restored.started_at == sp.started_at
    assert restored.ended_at == sp.ended_at


async def test_restored_user_span_data() -> None:
    # A user span made with span("name", **attrs) restores typed...
    async with ai.experimental_telemetry.span("turn", session="s1") as sp:
        pass
    restored = ai.experimental_telemetry.Span.model_validate(
        sp.model_dump(mode="json")
    )
    assert restored.data == ai.experimental_telemetry.CustomSpanData(
        attributes={"session": "s1"}
    )
    restored.set(extra=1)  # typed data means set() works after a restore

    # ...even when its name collides with a framework kind: the type
    # travels in the data, not the span name.
    async with ai.experimental_telemetry.span("loop_turn", foo=1) as sp2:
        pass
    collided = ai.experimental_telemetry.Span.model_validate(
        sp2.model_dump(mode="json")
    )
    assert collided.data == ai.experimental_telemetry.CustomSpanData(
        attributes={"foo": 1}
    )

    # ...while a user-defined span data type stays a dict on a bare
    # restore (its type isn't in the framework's tagged union);
    # parametrized validation returns the typed form.
    class RetrievalSpanData(pydantic.BaseModel):
        kind: Literal["retrieval"] = "retrieval"
        query: str

    async with ai.experimental_telemetry.span(
        RetrievalSpanData(query="q")
    ) as sp3:
        pass
    payload = sp3.model_dump(mode="json")
    bare = ai.experimental_telemetry.Span.model_validate(payload)
    assert isinstance(bare.data, dict)
    assert bare.data == {"kind": "retrieval", "query": "q"}
    typed = ai.experimental_telemetry.Span[RetrievalSpanData].model_validate(
        payload
    )
    assert typed.data == RetrievalSpanData(query="q")

    # A plain dataclass with a ``kind`` ClassVar also works as span
    # data; the ClassVar isn't serialized, so its dump has no tag.
    @dataclasses.dataclass
    class LegacySpanData:
        query: str

        kind: ClassVar[str] = "legacy"

    async with ai.experimental_telemetry.span(LegacySpanData("q")) as sp4:
        pass
    assert sp4.name == "legacy"
    bare = ai.experimental_telemetry.Span.model_validate(
        sp4.model_dump(mode="json")
    )
    assert isinstance(bare.data, dict)
    assert bare.data == {"query": "q"}


# ── create_span + push ────────────────────────────────────────────


async def test_create_span_reports_nothing(recorder: Recorder) -> None:
    sp = ai.experimental_telemetry.create_span("quiet")
    assert sp.started_at is None
    assert sp.ended_at is None
    # Even a push reports nothing before the span started.
    await sp.push()
    assert recorder.started == []
    assert recorder.ended == []


async def test_push_lifecycle_split_across_pushes(
    recorder: Recorder,
) -> None:
    sp = ai.experimental_telemetry.create_span("turn", session="s1")
    sp.started_at = ai.experimental_telemetry.now_ns()
    await sp.push()
    assert [s.name for s in recorder.started] == ["turn"]
    assert recorder.ended == []

    sp.ended_at = ai.experimental_telemetry.now_ns()
    await sp.push()
    (ended,) = recorder.ended
    assert ended.name == "turn"
    assert ended.ended_at == sp.ended_at


async def test_push_snapshots_are_frozen(recorder: Recorder) -> None:
    collector = ai.experimental_telemetry.Collector()
    data = ai.experimental_telemetry.ToolExecutionSpanData(
        tool_name="t", tool_call_id="tc", args={"x": 1}
    )
    with ai.experimental_telemetry.use_sink(collector):
        sp = ai.experimental_telemetry.create_span(data)
        sp.started_at = ai.experimental_telemetry.now_ns()
        await sp.push()
        # Mutations after a push don't leak into the snapshot.
        sp.data.args = {"x": 2}
    snapshot = collector.spans[sp.id]
    assert isinstance(
        snapshot.data, ai.experimental_telemetry.ToolExecutionSpanData
    )
    assert snapshot.data.args == {"x": 1}


async def test_finished_span_delivered_whole(recorder: Recorder) -> None:
    order: list[str] = []

    class Adapter:
        def on_span_start(self, span: ai.experimental_telemetry.Span) -> None:
            order.append(f"start {span.name}")

        def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            order.append(f"event {event.name}")

        def on_span_end(self, span: ai.experimental_telemetry.Span) -> None:
            order.append(f"end {span.name} error={span.error is not None}")

    # A span that lived elsewhere arrives as one complete record...
    sp = ai.experimental_telemetry.create_span("done-elsewhere")
    sp.started_at = ai.experimental_telemetry.now_ns()
    sp.events.append(
        ai.experimental_telemetry.SpanEvent(
            name="milestone",
            time_ns=ai.experimental_telemetry.now_ns(),
            attributes={},
        )
    )
    sp.ended_at = ai.experimental_telemetry.now_ns()
    sp.error = ai.experimental_telemetry.SpanError(type="E", message="m")
    payload = sp.model_dump(mode="json")

    adapter = Adapter()
    async with _registered(adapter):
        # ...and one push fires the full callback sequence.
        await ai.experimental_telemetry.Span.model_validate(payload).push()

    assert order == [
        "start done-elsewhere",
        "event milestone",
        "end done-elsewhere error=True",
    ]


async def test_adapter_view_updated_in_place() -> None:
    starts: list[ai.experimental_telemetry.Span] = []
    ends: list[ai.experimental_telemetry.Span] = []

    class Holder:
        def on_span_start(self, span: ai.experimental_telemetry.Span) -> None:
            starts.append(span)

        def on_span_end(self, span: ai.experimental_telemetry.Span) -> None:
            ends.append(span)

    async with _registered(Holder()):
        async with ai.experimental_telemetry.span("s") as sp:
            sp.set(a=1)

    # The adapter holds one object across callbacks; by span end it
    # shows the final data, like the live object it used to be handed.
    (start_view,) = starts
    (end_view,) = ends
    assert start_view is end_view
    assert isinstance(end_view.data, ai.experimental_telemetry.CustomSpanData)
    assert end_view.data.attributes == {"a": 1}
    assert end_view.ended_at is not None


async def test_repush_after_end_redelivers(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("s") as sp:
        pass
    assert [s.name for s in recorder.ended] == ["s"]
    # Re-pushing a completed span re-delivers it whole (the durable
    # re-emission path); dedup belongs to the backend, keyed on id.
    await sp.push()
    assert [s.name for s in recorder.started] == ["s", "s"]
    assert [s.name for s in recorder.ended] == ["s", "s"]
    assert recorder.ended[0].id == recorder.ended[1].id


# ── Sinks ─────────────────────────────────────────────────────────


async def test_use_sink_reroutes_pushes(recorder: Recorder) -> None:
    collector = ai.experimental_telemetry.Collector()
    with ai.experimental_telemetry.use_sink(collector):
        async with ai.experimental_telemetry.span("inside") as sp:
            sp.events.append(
                ai.experimental_telemetry.SpanEvent(
                    name="milestone",
                    time_ns=ai.experimental_telemetry.now_ns(),
                    attributes={},
                )
            )
            await sp.push()
    # Adapters saw nothing; the collector kept the latest snapshot.
    assert recorder.started == []
    assert recorder.ended == []
    (snapshot,) = collector.spans.values()
    assert snapshot.name == "inside"
    assert snapshot.ended_at is not None
    assert [e.name for e in snapshot.events] == ["milestone"]
    # Outside the context pushes reach adapters again.
    async with ai.experimental_telemetry.span("outside"):
        pass
    assert [s.name for s in recorder.ended] == ["outside"]


async def test_collector_ships_to_adapters_exactly_as_pushed(
    recorder: Recorder,
) -> None:
    # The durable-body pattern: collect inside, re-push from a "step".
    collector = ai.experimental_telemetry.Collector()
    with ai.experimental_telemetry.use_sink(collector):
        async with ai.experimental_telemetry.span("outer"):
            async with ai.experimental_telemetry.span("inner"):
                pass
    payload = [s.model_dump(mode="json") for s in collector.spans.values()]

    for item in payload:
        await ai.experimental_telemetry.Span.model_validate(item).push()

    assert {s.name for s in recorder.ended} == {"outer", "inner"}
    inner = next(s for s in recorder.ended if s.name == "inner")
    outer = next(s for s in recorder.ended if s.name == "outer")
    assert inner.parent_id == outer.id
    assert inner.trace_id == outer.trace_id


async def test_push_never_raises(recorder: Recorder) -> None:
    class BrokenSink:
        async def emit(self, span: ai.experimental_telemetry.Span) -> None:
            raise RuntimeError("sink bug")

    with ai.experimental_telemetry.use_sink(BrokenSink()):
        async with ai.experimental_telemetry.span("s"):
            pass  # both pushes hit the broken sink; neither raises


async def test_flush_reaches_sink_and_adapters() -> None:
    flushed: list[str] = []

    class FlushingAdapter:
        def flush(self) -> None:
            flushed.append("adapter")

    class FlushingSink:
        async def emit(self, span: ai.experimental_telemetry.Span) -> None:
            pass

        async def flush(self) -> None:
            flushed.append("sink")

    async with _registered(FlushingAdapter()):
        with ai.experimental_telemetry.use_sink(FlushingSink()):
            await ai.experimental_telemetry.flush()
    assert flushed == ["sink", "adapter"]


# ── span/trace ids ────────────────────────────────────────────────


async def test_ids_deterministic_under_use_random() -> None:
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
            await _add_event(outer, "milestone")

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
            await _add_event(sp, "e")
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


async def test_wrap_span_reads_error_after_loop() -> None:
    seen: list[ai.experimental_telemetry.SpanError | None] = []

    @ai.experimental_telemetry.wrap_span
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass
        # A failed span ends the loop like any other: the failure is
        # data on the span, never a live exception (the span may have
        # failed in another process).
        seen.append(span.error)

    error = ValueError("boom")
    async with _registered(adapter):
        with pytest.raises(ValueError) as excinfo:
            async with ai.experimental_telemetry.span("failing"):
                raise error
        # The app still gets the original exception...
        assert excinfo.value is error
        # ...and the bridge saw its serializable record.
        assert seen == [
            ai.experimental_telemetry.SpanError(
                type="ValueError", message="boom"
            )
        ]

        async with ai.experimental_telemetry.span("fine"):
            pass
        assert seen[-1] is None


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
            await _add_event(sp, "one")
            await _add_event(sp, "two")
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
        while (ev := (yield)) is not None:
            seen.append(ev.name)
        if span.error is not None:
            seen.append(f"error {span.error.message}")

    async with _registered(adapter):
        with pytest.raises(ValueError, match="boom"):
            async with ai.experimental_telemetry.span("failing") as sp:
                await _add_event(sp, "one")
                raise ValueError("boom")

    # Events delivered live, the error read after the loop ends.
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
            await _add_event(sp, "one")
            await _add_event(sp, "two")

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
            await _add_event(sp, "one")
            await _add_event(sp, "two")  # generator already dead: skipped

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
        order.append(f"end events={[e.name for e in span.events]}")

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("s") as sp:
            await _add_event(sp, "one")
            order.append("event added")

    assert order == ["start", "event added", "end events=['one']"]


# ── span events ───────────────────────────────────────────────────


async def test_events_appended_and_stamped() -> None:
    async with ai.experimental_telemetry.span("s") as sp:
        first = await _add_event(sp, "first", a=1)
        second = await _add_event(sp, "second")

    assert sp.events == [first, second]
    assert first.name == "first"
    assert first.attributes == {"a": 1}
    assert second.attributes == {}
    # Stamps from the ambient clock, in append order.
    assert sp.started_at is not None
    assert sp.ended_at is not None
    assert sp.started_at <= first.time_ns <= second.time_ns
    assert second.time_ns <= sp.ended_at


async def test_unpushed_events_delivered_by_end_push(
    recorder: Recorder,
) -> None:
    seen: list[str] = []

    class EventAdapter:
        def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            seen.append(event.name)

    async with _registered(EventAdapter()):
        async with ai.experimental_telemetry.span("s") as sp:
            # Appended but never pushed: the context manager's end
            # push carries it — nothing is lost, just delivered late.
            sp.events.append(
                ai.experimental_telemetry.SpanEvent(
                    name="quiet",
                    time_ns=ai.experimental_telemetry.now_ns(),
                    attributes={},
                )
            )

    assert seen == ["quiet"]
    (ended,) = recorder.ended
    assert [e.name for e in ended.events] == ["quiet"]


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
            await _add_event(sp, "milestone")

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
            event = await _add_event(sp, "milestone")

    assert seen == ["milestone"]
    assert sp.events == [event]
    assert [s.name for s in recorder.ended] == ["s"]


# ── use_clock ─────────────────────────────────────────────────────


def _ticking_clock(now_ns: int, tick_ns: int = 10) -> Callable[[], int]:
    """Deterministic clock: each reading ticks forward by ``tick_ns``."""

    def time_ns() -> int:
        nonlocal now_ns
        now_ns += tick_ns
        return now_ns

    return time_ns


async def _stamps() -> tuple[int | None, int, int | None]:
    async with ai.experimental_telemetry.span("s") as sp:
        event = await _add_event(sp, "milestone")
    return sp.started_at, event.time_ns, sp.ended_at


async def test_use_clock_overrides_and_restores() -> None:
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
    assert sp.started_at is not None
    assert sp.started_at > 1_030


async def test_use_clock_decorator_handles_async_functions() -> None:
    # Works as a decorator on an async fn; the clock is shared across
    # calls, so the second call continues where the first left off.
    @ai.experimental_telemetry.use_clock(_ticking_clock(1_000))
    async def run() -> tuple[int | None, int, int | None]:
        return await _stamps()

    assert await run() == (1_010, 1_020, 1_030)
    assert await run() == (1_040, 1_050, 1_060)


async def test_use_clock_accepts_time_time_ns() -> None:
    before = time.time_ns()
    with ai.experimental_telemetry.use_clock(time.time_ns):
        async with ai.experimental_telemetry.span("s") as sp:
            pass
    assert sp.started_at is not None
    assert sp.ended_at is not None
    assert before <= sp.started_at <= sp.ended_at <= time.time_ns()
