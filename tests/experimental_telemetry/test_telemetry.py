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
    **attrs: Any,
) -> ai.experimental_telemetry.SpanEvent:
    """The event pattern: append, then push for live delivery."""
    event = sp.add_event(name, **attrs)
    await sp.push()
    return event


@contextlib.asynccontextmanager
async def _registered(
    adapter: ai.experimental_telemetry.AdapterProtocol
    | ai.experimental_telemetry.AdapterCallable,
) -> AsyncIterator[None]:
    ai.experimental_telemetry.register(adapter)
    try:
        yield
    finally:
        ai.experimental_telemetry.unregister(adapter)


class _NoopAdapter:
    """Raw-hook adapter base: tests override the hooks they care about."""

    async def on_span_start(self, span: ai.experimental_telemetry.Span) -> None:
        pass

    async def on_span_event(
        self,
        span: ai.experimental_telemetry.Span,
        event: ai.experimental_telemetry.SpanEvent,
    ) -> None:
        pass

    async def on_span_end(self, span: ai.experimental_telemetry.Span) -> None:
        pass


async def test_nesting_and_ids(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("outer") as outer:
        assert ai.experimental_telemetry.current_span() is outer
        async with ai.experimental_telemetry.span("inner") as inner:
            assert inner.parent_id == outer.id
            assert inner.trace_id == outer.trace_id
    assert ai.experimental_telemetry.current_span() is None
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


def test_embed_span_roundtrips() -> None:
    data = ai.experimental_telemetry.EmbedSpanData(
        model="m",
        provider="p",
        input_count=1,
        output_count=1,
        dimensions=8,
    )
    span = ai.experimental_telemetry.Span(
        name="embed", data=data, id="span-1", trace_id="trace-1"
    )

    restored = ai.experimental_telemetry.Span.model_validate(
        span.model_dump(mode="json")
    )

    assert isinstance(restored.data, ai.experimental_telemetry.EmbedSpanData)
    assert restored.data.dimensions == 8


async def test_set_attrs(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("s") as span:
        span.set_attrs(a=1)
        span.set_attrs(b=2)
    (ended,) = recorder.ended
    assert isinstance(ended.data, ai.experimental_telemetry.CustomSpanData)
    assert ended.data.attrs == {"a": 1, "b": 2}


async def test_set_attrs_rejects_framework_spans() -> None:
    async with ai.experimental_telemetry.span(
        ai.experimental_telemetry.LoopTurnSpanData()
    ) as span:
        with pytest.raises(TypeError):
            span.set_attrs(a=1)


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
            assert ai.experimental_telemetry.current_span() is outer
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
    class Broken(_NoopAdapter):
        async def on_span_start(
            self, span: ai.experimental_telemetry.Span
        ) -> None:
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

    class AsyncAdapter(_NoopAdapter):
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


async def test_trace_attrs_propagate_to_children(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("root") as root:
        root.trace_attrs["session.id"] = "s-1"
        async with ai.experimental_telemetry.span("child") as child:
            # A copy of the parent's attrs at creation time...
            assert child.trace_attrs == {"session.id": "s-1"}
            async with ai.experimental_telemetry.span("grandchild") as gc:
                assert gc.trace_attrs == {"session.id": "s-1"}
            # ...independent of the parent's dict.
            child.trace_attrs["extra"] = True
        assert root.trace_attrs == {"session.id": "s-1"}
        # Later mutations reach spans created afterwards, not before.
        root.trace_attrs["user.id"] = "u-1"
        async with ai.experimental_telemetry.span("sibling") as sibling:
            assert sibling.trace_attrs == {
                "session.id": "s-1",
                "user.id": "u-1",
            }
        assert child.trace_attrs == {"session.id": "s-1", "extra": True}


async def test_trace_attrs_survive_restore(recorder: Recorder) -> None:
    # "Process one": trace_attrs travel inside the serialized span.
    async with ai.experimental_telemetry.span("origin") as origin:
        origin.trace_attrs["session.id"] = "s-1"
        payload = origin.model_dump(mode="json")

    # "Process two": children of the restored span inherit them.
    restored = ai.experimental_telemetry.Span.model_validate(payload)
    assert restored.trace_attrs == {"session.id": "s-1"}
    async with ai.experimental_telemetry.span(
        "pickup", parent=restored
    ) as pickup:
        assert pickup.trace_attrs == {"session.id": "s-1"}


async def test_use_span_parents_without_lifecycle(
    recorder: Recorder,
) -> None:
    # ``use_span`` is pure context plumbing: no pushes, no timestamps.
    outer = ai.experimental_telemetry.create_span("outer")
    async with ai.experimental_telemetry.use_span(outer):
        assert ai.experimental_telemetry.current_span() is outer
        async with ai.experimental_telemetry.span("child") as child:
            assert child.parent_id == outer.id
            assert child.trace_id == outer.trace_id
    assert ai.experimental_telemetry.current_span() is None
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
    # A user span made with span("name") restores typed...
    async with ai.experimental_telemetry.span("turn") as sp:
        sp.set_attrs(session="s1")
    restored = ai.experimental_telemetry.Span.model_validate(
        sp.model_dump(mode="json")
    )
    assert restored.data == ai.experimental_telemetry.CustomSpanData(
        attrs={"session": "s1"}
    )
    restored.set_attrs(extra=1)

    # ...even when its name collides with a framework kind: the type
    # travels in the data, not the span name.
    async with ai.experimental_telemetry.span("loop_turn") as sp2:
        sp2.set_attrs(foo=1)
    collided = ai.experimental_telemetry.Span.model_validate(
        sp2.model_dump(mode="json")
    )
    assert collided.data == ai.experimental_telemetry.CustomSpanData(
        attrs={"foo": 1}
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
    sp = ai.experimental_telemetry.create_span("turn")
    sp.set_attrs(session="s1")
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
    sink = ai.experimental_telemetry.DictSink()
    data = ai.experimental_telemetry.ToolExecutionSpanData(
        tool_name="t", tool_call_id="tc", args={"x": 1}
    )
    async with ai.experimental_telemetry.use_sink(sink):
        sp = ai.experimental_telemetry.create_span(data)
        sp.started_at = ai.experimental_telemetry.now_ns()
        await sp.push()
        # Mutations after a push don't leak into the snapshot.
        sp.data.args = {"x": 2}
    snapshot = sink.spans[sp.id]
    assert isinstance(
        snapshot.data, ai.experimental_telemetry.ToolExecutionSpanData
    )
    assert snapshot.data.args == {"x": 1}


async def test_finished_span_delivered_whole(recorder: Recorder) -> None:
    order: list[str] = []

    class Adapter:
        async def on_span_start(
            self, span: ai.experimental_telemetry.Span
        ) -> None:
            order.append(f"start {span.name}")

        async def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            order.append(f"event {event.name}")

        async def on_span_end(
            self, span: ai.experimental_telemetry.Span
        ) -> None:
            order.append(f"end {span.name} error={span.error is not None}")

    # A span that lived elsewhere arrives as one complete record...
    sp = ai.experimental_telemetry.create_span("done-elsewhere")
    sp.started_at = ai.experimental_telemetry.now_ns()
    sp.events.append(
        ai.experimental_telemetry.SpanEvent(
            name="milestone",
            time_ns=ai.experimental_telemetry.now_ns(),
            attrs={},
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

    class Holder(_NoopAdapter):
        async def on_span_start(
            self, span: ai.experimental_telemetry.Span
        ) -> None:
            starts.append(span)

        async def on_span_end(
            self, span: ai.experimental_telemetry.Span
        ) -> None:
            ends.append(span)

    async with _registered(Holder()):
        async with ai.experimental_telemetry.span("s") as sp:
            sp.set_attrs(a=1)

    # The adapter holds one object across callbacks; by span end it
    # shows the final data, like the live object it used to be handed.
    (start_view,) = starts
    (end_view,) = ends
    assert start_view is end_view
    assert isinstance(end_view.data, ai.experimental_telemetry.CustomSpanData)
    assert end_view.data.attrs == {"a": 1}
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
    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        async with ai.experimental_telemetry.span("inside") as sp:
            sp.events.append(
                ai.experimental_telemetry.SpanEvent(
                    name="milestone",
                    time_ns=ai.experimental_telemetry.now_ns(),
                    attrs={},
                )
            )
            await sp.push()
    # Adapters saw nothing; the sink kept the latest snapshot.
    assert recorder.started == []
    assert recorder.ended == []
    (snapshot,) = sink.spans.values()
    assert snapshot.name == "inside"
    assert snapshot.ended_at is not None
    assert [e.name for e in snapshot.events] == ["milestone"]
    # Outside the context pushes reach adapters again.
    async with ai.experimental_telemetry.span("outside"):
        pass
    assert [s.name for s in recorder.ended] == ["outside"]


async def test_use_sink_accepts_none(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.use_sink(None):
        async with ai.experimental_telemetry.span("inside"):
            pass
    assert [s.name for s in recorder.ended] == ["inside"]


async def test_use_sink_none_preserves_outer_sink(recorder: Recorder) -> None:
    sink = ai.experimental_telemetry.DictSink()
    async with (
        ai.experimental_telemetry.use_sink(sink),
        ai.experimental_telemetry.use_sink(None),
    ):
        async with ai.experimental_telemetry.span("inside"):
            pass
    assert [s.name for s in sink.finished_spans] == ["inside"]
    assert recorder.ended == []


async def test_dict_sink_ships_to_adapters_exactly_as_pushed(
    recorder: Recorder,
) -> None:
    # The durable-body pattern: collect inside, re-push from a "step".
    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        async with ai.experimental_telemetry.span("outer"):
            async with ai.experimental_telemetry.span("inner"):
                pass
    payload = [s.model_dump(mode="json") for s in sink.spans.values()]

    for item in payload:
        await ai.experimental_telemetry.Span.model_validate(item).push()

    assert {s.name for s in recorder.ended} == {"outer", "inner"}
    inner = next(s for s in recorder.ended if s.name == "inner")
    outer = next(s for s in recorder.ended if s.name == "outer")
    assert inner.parent_id == outer.id
    assert inner.trace_id == outer.trace_id


async def test_push_never_raises(recorder: Recorder) -> None:
    class BrokenSink:
        async def on_push(self, span: ai.experimental_telemetry.Span) -> None:
            raise RuntimeError("sink bug")

    async with ai.experimental_telemetry.use_sink(BrokenSink()):
        async with ai.experimental_telemetry.span("s"):
            pass  # both pushes hit the broken sink; neither raises


# ── span/trace ids ────────────────────────────────────────────────


async def test_ids_deterministic_under_use_random() -> None:
    async def run() -> tuple[str, str, str]:
        async with ai.messages.use_random(random.Random(7)):
            async with ai.experimental_telemetry.span("outer") as outer:
                async with ai.experimental_telemetry.span("inner") as inner:
                    pass
        return outer.trace_id, outer.id, inner.id

    # The replay contract: re-running the same work under the same
    # random source re-emits spans with identical identities.
    assert await run() == await run()


# ── @adapter on a class ──────────────────────────────────────────


async def test_adapter_class_driven() -> None:
    @ai.experimental_telemetry.adapter
    class Vendor:
        def __init__(self) -> None:
            self.log: list[str] = []  # vendor-specific state

        async def __call__(
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

    # The mixed-in driver ran one generator frame per span, with the
    # same semantics as decorating a free-standing function.
    assert vendor.log == [
        "start outer",
        "start inner",
        "end inner",
        "event milestone",
        "end outer",
    ]


async def test_adapter_class_keeps_identity() -> None:
    @ai.experimental_telemetry.adapter
    class Vendor:
        """Vendor bridge."""

        async def __call__(
            self, span: ai.experimental_telemetry.Span
        ) -> AsyncGenerator[None, Any]:
            while (yield) is not None:
                pass

    assert Vendor.__name__ == "Vendor"
    assert Vendor.__qualname__.endswith("Vendor")
    assert Vendor.__doc__ == "Vendor bridge."
    assert isinstance(Vendor(), ai.experimental_telemetry.AdapterMixin)


async def test_adapter_decorated_class_supports_subclassing() -> None:
    log: list[str] = []

    @ai.experimental_telemetry.adapter
    class Base:
        def label(self, span: ai.experimental_telemetry.Span) -> str:
            return span.name

        async def __call__(
            self, span: ai.experimental_telemetry.Span
        ) -> AsyncGenerator[None, Any]:
            while (yield) is not None:
                pass
            log.append(f"end {self.label(span)}")

    class Enriched(Base):
        def label(self, span: ai.experimental_telemetry.Span) -> str:
            return f"enriched:{span.name}"

    # A subclass of a decorated class inherits the driver; no second
    # decoration needed.
    async with _registered(Enriched()):
        async with ai.experimental_telemetry.span("s"):
            pass

    assert log == ["end enriched:s"]


async def test_adapter_class_requires_async_generator_call() -> None:
    class NoCall:
        pass

    class PlainCall:
        async def __call__(self, span: ai.experimental_telemetry.Span) -> None:
            pass

    bad_classes: list[Any] = [NoCall, PlainCall]
    for bad in bad_classes:
        with pytest.raises(TypeError, match="async generator"):
            ai.experimental_telemetry.adapter(bad)


async def test_adapter_rejects_double_decoration() -> None:
    @ai.experimental_telemetry.adapter
    class Vendor:
        async def __call__(
            self, span: ai.experimental_telemetry.Span
        ) -> AsyncGenerator[None, Any]:
            while (yield) is not None:
                pass

    with pytest.raises(TypeError, match="already an adapter"):
        ai.experimental_telemetry.adapter(Vendor)


async def test_register_rejects_undecorated_class() -> None:
    class Vendor:
        async def __call__(
            self, span: ai.experimental_telemetry.Span
        ) -> AsyncGenerator[None, Any]:
            while (yield) is not None:
                pass

    with pytest.raises(TypeError, match="@adapter"):
        ai.experimental_telemetry.register(Vendor())


async def test_adapter_defaults_are_noops(recorder: Recorder) -> None:
    # A bare AdapterMixin (its defaults) is valid: no per-span
    # frames.
    async with _registered(ai.experimental_telemetry.AdapterMixin()):
        async with ai.experimental_telemetry.span("s") as sp:
            await _add_event(sp, "e")
    assert [s.name for s in recorder.ended] == ["s"]


async def test_hook_override_composes_with_super() -> None:
    log: list[str] = []

    @ai.experimental_telemetry.adapter
    class Both:
        async def __call__(
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
            # super() plugs it back in (the mixin sits after this
            # class in the MRO; static checkers can't see it).
            await super().on_span_end(  # type: ignore[misc]  # ty: ignore[unresolved-attribute]
                span
            )

    async with _registered(Both()):
        async with ai.experimental_telemetry.span("s"):
            pass

    assert log == ["hook end s", "frame end s"]


async def test_decorated_class_state_cannot_collide_with_driver() -> None:
    @ai.experimental_telemetry.adapter
    class Clashy:
        def __init__(self) -> None:
            self._live = "mine"  # same name the driver mangles away
            self.ended: list[str] = []

        async def __call__(
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


async def test_adapter_function_returns_adapter() -> None:
    @ai.experimental_telemetry.adapter
    async def vendor(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass

    assert isinstance(vendor, ai.experimental_telemetry.AdapterMixin)
    assert "adapter" in repr(vendor)


# ── @adapter on a function ────────────────────────────────────────


async def test_adapter_locals_across_yield() -> None:
    events: list[str] = []

    @ai.experimental_telemetry.adapter
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        name = span.name  # local state, no bookkeeping dict
        events.append(f"start {name}")
        while (yield) is not None:
            pass
        assert span.ended_at is not None
        assert isinstance(span.data, ai.experimental_telemetry.CustomSpanData)
        events.append(f"end {name} a={span.data.attrs.get('a')}")

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("outer") as outer:
            async with ai.experimental_telemetry.span("inner") as inner:
                inner.set_attrs(a=1)
            outer.set_attrs(a=2)

    # One generator frame per live span, each resumed at its own
    # span's end with the final data visible after the yield.
    assert events == [
        "start outer",
        "start inner",
        "end inner a=1",
        "end outer a=2",
    ]


async def test_adapter_reads_error_after_loop() -> None:
    seen: list[ai.experimental_telemetry.SpanError | None] = []

    @ai.experimental_telemetry.adapter
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


async def test_adapter_opt_out_before_yield(recorder: Recorder) -> None:
    ended: list[str] = []

    @ai.experimental_telemetry.adapter
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


async def test_adapter_failures_isolated(recorder: Recorder) -> None:
    @ai.experimental_telemetry.adapter
    async def broken_before(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        raise RuntimeError("pre-yield bug")
        while (yield) is not None:
            pass

    @ai.experimental_telemetry.adapter
    async def broken_after(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        while (yield) is not None:
            pass
        raise RuntimeError("post-loop bug")

    @ai.experimental_telemetry.adapter
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


async def test_adapter_rejects_plain_functions() -> None:
    async def not_a_generator(span: ai.experimental_telemetry.Span) -> None:
        pass

    fn: Any = not_a_generator
    with pytest.raises(TypeError, match="async generator function"):
        ai.experimental_telemetry.adapter(fn)


async def test_adapter_events_live_loop() -> None:
    seen: list[str] = []

    @ai.experimental_telemetry.adapter
    async def adapter(
        span: ai.experimental_telemetry.Span,
    ) -> AsyncGenerator[None, Any]:
        vendor = f"vendor:{span.name}"  # frame-local state
        while (ev := (yield)) is not None:
            # Delivered live, while the span is still open.
            assert span.ended_at is None
            seen.append(f"{vendor} event {ev.name}")
        assert isinstance(span.data, ai.experimental_telemetry.CustomSpanData)
        seen.append(f"{vendor} end a={span.data.attrs.get('a')}")

    async with _registered(adapter):
        async with ai.experimental_telemetry.span("s") as sp:
            await _add_event(sp, "one")
            await _add_event(sp, "two")
            sp.set_attrs(a=1)

    assert seen == [
        "vendor:s event one",
        "vendor:s event two",
        "vendor:s end a=1",
    ]


async def test_adapter_error_reaches_loop() -> None:
    seen: list[str] = []

    @ai.experimental_telemetry.adapter
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


async def test_adapter_early_finish_opts_out(
    recorder: Recorder,
) -> None:
    seen: list[str] = []
    ended: list[str] = []

    @ai.experimental_telemetry.adapter
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


async def test_adapter_raising_event_handler_isolated(
    recorder: Recorder,
) -> None:
    @ai.experimental_telemetry.adapter
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


async def test_adapter_drain_loop() -> None:
    order: list[str] = []

    @ai.experimental_telemetry.adapter
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


async def test_events_appended_and_stamped(recorder: Recorder) -> None:
    async with ai.experimental_telemetry.span("s") as sp:
        first = await _add_event(sp, "first", a=1)
        second = await _add_event(sp, "second")

    assert sp.events == [first, second]
    assert first.name == "first"
    assert first.attrs == {"a": 1}
    assert second.attrs == {}
    # Stamps from the ambient clock, in append order.
    assert sp.started_at is not None
    assert sp.ended_at is not None
    assert sp.started_at <= first.time_ns <= second.time_ns
    assert second.time_ns <= sp.ended_at


async def test_unpushed_events_delivered_by_end_push(
    recorder: Recorder,
) -> None:
    seen: list[str] = []

    class EventAdapter(_NoopAdapter):
        async def on_span_event(
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
                    attrs={},
                )
            )

    assert seen == ["quiet"]
    (ended,) = recorder.ended
    assert [e.name for e in ended.events] == ["quiet"]


async def test_span_event_dispatched_live_to_multiple_adapters(
    recorder: Recorder,
) -> None:
    seen: list[tuple[str, str, bool]] = []

    class FirstAdapter(_NoopAdapter):
        async def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            seen.append(("first", event.name, span.ended_at is None))

    class SecondAdapter(_NoopAdapter):
        async def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            seen.append(("second", event.name, span.ended_at is None))

    async with _registered(FirstAdapter()), _registered(SecondAdapter()):
        async with ai.experimental_telemetry.span("s") as sp:
            await _add_event(sp, "milestone")

    # Both handlers saw the event while the span was still live; the
    # recorder (no-op on_span_event) stayed out of the way.
    assert seen == [
        ("first", "milestone", True),
        ("second", "milestone", True),
    ]
    assert [s.name for s in recorder.ended] == ["s"]


async def test_span_event_raising_handler_isolated(
    recorder: Recorder,
) -> None:
    seen: list[str] = []

    class Broken(_NoopAdapter):
        async def on_span_event(
            self,
            span: ai.experimental_telemetry.Span,
            event: ai.experimental_telemetry.SpanEvent,
        ) -> None:
            raise RuntimeError("adapter bug")

    class Fine(_NoopAdapter):
        async def on_span_event(
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


# ── noop spans (telemetry off) ───────────────────────────────────


async def test_span_without_adapters_is_noop() -> None:
    async with ai.experimental_telemetry.span("outer") as sp:
        assert sp.id == ""
        assert sp.trace_id == ""
        assert sp.parent_id is None
        assert sp.started_at is None
        # Never current: nothing to parent under.
        assert ai.experimental_telemetry.current_span() is None
        # Attribute writes still work; they are just never observed.
        sp.set_attrs(b=2)
        event = sp.add_event("milestone", c=3)
    assert event.name == "milestone"
    assert event.time_ns == 0
    assert event.attrs == {"c": 3}
    assert sp.events == [event]
    assert sp.ended_at is None


async def test_noop_span_reraises_without_recording() -> None:
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
    # non-deterministic calls, so no use_time setup is needed.
    async with ai.experimental_telemetry.span("outer") as sp:
        sp.add_event("milestone")
        await sp.push()
        async with ai.experimental_telemetry.span(
            ai.experimental_telemetry.LoopTurnSpanData()
        ):
            pass


async def test_no_rng_draws_without_adapters() -> None:
    async def id_after_spans(spans: int) -> str:
        async with ai.messages.use_random(random.Random(7)):
            for _ in range(spans):
                async with ai.experimental_telemetry.span("s"):
                    pass
            return ai.messages.generate_id()

    # Noop spans consume nothing from the ambient random source, so
    # ids drawn afterwards are unaffected by how many spans opened.
    assert await id_after_spans(0) == await id_after_spans(5)


async def test_routed_sink_keeps_spans_live_without_adapters() -> None:
    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        async with ai.experimental_telemetry.span("s") as sp:
            pass
    assert sp.id
    assert [s.name for s in sink.finished_spans] == ["s"]


# ── use_time ──────────────────────────────────────────────────────


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


async def test_use_time_overrides_and_restores(recorder: Recorder) -> None:
    # A ticking clock gives a deterministic timestamp sequence; the
    # override drives started_at, event stamps, and ended_at alike.
    async with ai.experimental_telemetry.use_time(_ticking_clock(1_000)):
        first = await _stamps()
    async with ai.experimental_telemetry.use_time(_ticking_clock(1_000)):
        second = await _stamps()

    assert first == second == (1_010, 1_020, 1_030)

    # Restored on exit -- back to the wall clock.
    async with ai.experimental_telemetry.span("s") as sp:
        pass
    assert sp.started_at is not None
    assert sp.started_at > 1_030


async def test_use_time_decorator_handles_async_functions(
    recorder: Recorder,
) -> None:
    # Works as a decorator on an async fn; the clock is shared across
    # calls, so the second call continues where the first left off.
    @ai.experimental_telemetry.use_time(_ticking_clock(1_000))
    async def run() -> tuple[int | None, int, int | None]:
        return await _stamps()

    assert await run() == (1_010, 1_020, 1_030)
    assert await run() == (1_040, 1_050, 1_060)


async def test_use_time_accepts_time_time_ns(recorder: Recorder) -> None:
    before = time.time_ns()
    async with ai.experimental_telemetry.use_time(time.time_ns):
        async with ai.experimental_telemetry.span("s") as sp:
            pass
    assert sp.started_at is not None
    assert sp.ended_at is not None
    assert before <= sp.started_at <= sp.ended_at <= time.time_ns()


# ── Sugar: stamps, add_event, is_enabled, shipping helpers ────────


async def test_use_span_accepts_none(recorder: Recorder) -> None:
    # "no span" is a normal state in gated instrumentation; None means
    # no reparenting, no separate code path at the callsite.
    async with ai.experimental_telemetry.use_span(None):
        assert ai.experimental_telemetry.current_span() is None
        async with ai.experimental_telemetry.span("child") as child:
            pass
    assert child.parent_id is None


async def test_is_enabled_reflects_sinks_and_adapters() -> None:
    # No adapters registered here (no recorder fixture), default sink.
    assert not ai.experimental_telemetry.is_enabled()
    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        assert ai.experimental_telemetry.is_enabled()
    assert not ai.experimental_telemetry.is_enabled()


async def test_is_enabled_with_adapter_registered(recorder: Recorder) -> None:
    assert ai.experimental_telemetry.is_enabled()


async def test_stamp_start_and_end(recorder: Recorder) -> None:
    clock = _ticking_clock(100)
    async with ai.experimental_telemetry.use_time(clock):
        sp = ai.experimental_telemetry.create_span("turn").stamp_start()
        assert sp.started_at == 110
        assert sp.stamp_end(error=ValueError("boom")) is sp
    assert sp.ended_at == 120
    assert sp.error == ai.experimental_telemetry.SpanError(
        type="ValueError", message="boom"
    )
    # Stamps only write fields; nothing was pushed.
    assert recorder.started == []

    # A SpanError passes through as-is; error=None keeps an existing one.
    err = ai.experimental_telemetry.SpanError(type="TurnError", message="m")
    sp2 = ai.experimental_telemetry.create_span("t").stamp_start()
    sp2.stamp_end(error=err)
    assert sp2.error is err
    sp2.stamp_end()
    assert sp2.error is err


async def test_add_event_appends_without_pushing() -> None:
    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        async with ai.experimental_telemetry.span("s") as sp:
            event = sp.add_event("cache_hit", {"cache.key": "k"}, size=3)
            assert event.attrs == {"cache.key": "k", "size": 3}
            assert sp.events == [event]
            # Append only: the latest snapshot (from the start push)
            # doesn't have it yet.
            assert sink.spans[sp.id].events == []
    # The end push delivered it.
    assert [e.name for e in sink.spans[sp.id].events] == ["cache_hit"]


async def test_dict_sink_finished_spans_and_push_all(
    recorder: Recorder,
) -> None:
    sink = ai.experimental_telemetry.DictSink()
    async with ai.experimental_telemetry.use_sink(sink):
        async with ai.experimental_telemetry.span("done"):
            pass
        dangling = ai.experimental_telemetry.create_span("open").stamp_start()
        await dangling.push()
    # Only complete records are safe to ship.
    assert [s.name for s in sink.finished_spans] == ["done"]

    # The ship step: re-deliver dumped payloads where the adapters are.
    payload = [s.model_dump(mode="json") for s in sink.finished_spans]
    await ai.experimental_telemetry.push_all(payload)
    assert [s.name for s in recorder.started] == ["done"]
    assert [s.name for s in recorder.ended] == ["done"]

    # Live spans are accepted alongside dumped ones.
    await ai.experimental_telemetry.push_all([dangling.stamp_end()])
    assert [s.name for s in recorder.ended] == ["done", "open"]


async def test_attribute_mappings_for_dotted_names(
    recorder: Recorder,
) -> None:
    # Viewer attribute names ("session.id") aren't valid keywords, so
    # set_attrs() merges a positional mapping with keywords.
    async with ai.experimental_telemetry.span("generate_title") as sp:
        sp.set_attrs({"session.id": "s1"}, model="haiku")
        sp.set_attrs({"output.value": "t"}, plain=1)
    assert sp.data.attrs == {
        "session.id": "s1",
        "model": "haiku",
        "output.value": "t",
        "plain": 1,
    }
