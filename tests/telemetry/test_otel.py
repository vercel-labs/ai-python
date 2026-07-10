"""Otel adapter: names, attributes, parenting, context bridging."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import ai
from ai.telemetry import otel

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def otel_env() -> Iterator[tuple[InMemorySpanExporter, TracerProvider]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = otel.install(tracer_provider=provider)
    yield exporter, provider
    ai.telemetry.unregister(adapter)


async def test_nesting_names_and_attributes(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    async with ai.telemetry.span("outer", foo="bar"):
        async with ai.telemetry.span("inner"):
            pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    outer, inner = spans["outer"], spans["inner"]
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id
    assert outer.attributes is not None
    assert outer.attributes["foo"] == "bar"


async def test_model_call_attributes(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    mock_llm([[text_msg("hello")]])
    async with ai.stream(MOCK_MODEL, [ai.user_message("hi")]) as stream:
        async for _ in stream:
            pass

    (span,) = exporter.get_finished_spans()
    assert span.name == "chat mock-model"
    assert span.attributes is not None
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == "mock-model"
    assert "hello" in str(span.attributes["gen_ai.output.messages"])


async def test_error_status(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    with pytest.raises(ValueError, match="boom"):
        async with ai.telemetry.span("failing"):
            raise ValueError("boom")

    (span,) = exporter.get_finished_spans()
    assert not span.status.is_ok
    events = [e.name for e in span.events]
    assert "exception" in events


async def test_tool_exception_recorded(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env

    @ai.tool
    async def boom() -> str:
        """Tool that always fails."""
        raise ValueError("nope")

    mock_llm([[tool_call_msg(name="boom")], [text_msg("done")]])
    my_agent = ai.Agent(tools=[boom])
    async with my_agent.run(MOCK_MODEL, [ai.user_message("go")]) as stream:
        async for _ in stream:
            pass

    (span,) = [
        s
        for s in exporter.get_finished_spans()
        if s.name == "execute_tool boom"
    ]
    # The tool exception is caught by the framework, but the span still
    # records it: ERROR status plus a standard otel exception event.
    assert not span.status.is_ok
    events = {e.name: e for e in span.events}
    assert "exception" in events
    attrs = events["exception"].attributes
    assert attrs is not None
    assert attrs["exception.type"] == "ValueError"
    assert attrs["exception.message"] == "nope"


async def test_span_events_exported_with_original_timestamps(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    marker = object()
    async with ai.telemetry.span("s") as sp:
        first = await sp.add_event("first_token", event_type="TextStart")
        second = await sp.add_event("custom", obj=marker)

    (span,) = exporter.get_finished_spans()
    events = {e.name: e for e in span.events}
    assert events["first_token"].timestamp == first.time_ns
    assert events["custom"].timestamp == second.time_ns
    assert events["first_token"].attributes is not None
    assert events["first_token"].attributes["event_type"] == "TextStart"
    # Non-primitive attribute values are sanitized to their repr.
    assert events["custom"].attributes is not None
    assert events["custom"].attributes["obj"] == repr(marker)


async def test_stream_milestones_exported(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    mock_llm([[text_msg("hello")]])
    async with ai.stream(MOCK_MODEL, [ai.user_message("hi")]) as stream:
        async for _ in stream:
            pass

    (span,) = exporter.get_finished_spans()
    assert [e.name for e in span.events] == [
        "first_token",
        "response_complete",
    ]


def _span_int(id_: str) -> int:
    return int(id_.rsplit("_", 1)[-1], 16)


async def test_our_ids_carried_through(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    async with ai.telemetry.span("outer") as outer:
        async with ai.telemetry.span("inner") as inner:
            pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    # The exported otel ids are our ids, so backends can dedupe and
    # cross-reference against saved state.
    assert spans["outer"].context.span_id == _span_int(outer.id)
    assert spans["outer"].context.trace_id == _span_int(outer.trace_id)
    assert spans["inner"].context.span_id == _span_int(inner.id)
    assert spans["inner"].context.trace_id == _span_int(outer.trace_id)


async def test_replay_spans_skipped_live_children_parented(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    async with ai.telemetry.span("replayed", replay=True) as replayed:
        async with ai.telemetry.span("live"):
            pass

    # Only the live span is exported, but it still hangs under the
    # replayed parent's reproducible ids.
    (span,) = exporter.get_finished_spans()
    assert span.name == "live"
    assert span.parent is not None
    assert span.parent.span_id == _span_int(replayed.id)
    assert span.context.trace_id == _span_int(replayed.trace_id)


async def test_export_replays_opt_in() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = otel.install(tracer_provider=provider, export_replays=True)
    try:
        async with ai.telemetry.span("replayed", replay=True):
            pass
    finally:
        ai.telemetry.unregister(adapter)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["ai.replay"] is True


async def test_cross_life_trace_converges(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env

    # Life one: the run starts live, completes one step, then dies.
    with ai.messages.use_random(random.Random(7)):
        async with ai.telemetry.span("run"):
            async with ai.telemetry.span("step"):
                pass

    # Life two: deterministic replay re-runs the same code (same ids),
    # then continues with live work.
    with ai.messages.use_random(random.Random(7)):
        async with ai.telemetry.span("run", replay=True) as run:
            async with ai.telemetry.span("step", replay=True):
                pass
            async with ai.telemetry.span("tail"):
                pass

    spans = exporter.get_finished_spans()
    names = sorted(s.name for s in spans)
    # No duplicates: replayed emissions were skipped.
    assert names == ["run", "step", "tail"]
    by_name = {s.name: s for s in spans}
    # The live tail from life two parents under the run span exported
    # in life one — the ids line up across process lives.
    assert by_name["run"].context.span_id == _span_int(run.id)
    tail = by_name["tail"]
    assert tail.parent is not None
    assert tail.parent.span_id == by_name["run"].context.span_id
    assert tail.context.trace_id == by_name["run"].context.trace_id


async def test_span_ref_parent_exported(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    async with ai.telemetry.span("origin") as origin:
        payload = origin.ref.model_dump()

    ref = ai.telemetry.SpanRef.model_validate(payload)
    async with ai.telemetry.span("pickup", parent=ref):
        pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    pickup, origin_otel = spans["pickup"], spans["origin"]
    assert pickup.parent is not None
    assert pickup.parent.span_id == origin_otel.context.span_id
    assert pickup.context.trace_id == origin_otel.context.trace_id


async def test_flush_reaches_provider(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, provider = otel_env
    calls: list[bool] = []
    original = provider.force_flush

    def force_flush(timeout_millis: int = 30000) -> bool:
        calls.append(True)
        return original(timeout_millis)

    monkeypatch.setattr(provider, "force_flush", force_flush)
    await ai.telemetry.flush()
    assert calls == [True]


async def test_raw_otel_span_nests_under_ours(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, provider = otel_env
    tracer = provider.get_tracer("test")
    async with ai.telemetry.span("outer"):
        with tracer.start_as_current_span("raw"):
            pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    raw, outer = spans["raw"], spans["outer"]
    assert raw.parent is not None
    assert raw.parent.span_id == outer.context.span_id


async def test_non_current_span_not_attached_to_otel_context(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, provider = otel_env
    tracer = provider.get_tracer("test")
    async with ai.telemetry.span("outer"):
        async with ai.telemetry.span("overlapping", set_as_current=False):
            # A raw otel span opened while the non-current span is
            # open parents like our own spans do: under the outer
            # span, not under the overlapping one.
            with tracer.start_as_current_span("raw"):
                pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    outer, overlapping, raw = (
        spans["outer"],
        spans["overlapping"],
        spans["raw"],
    )
    assert raw.parent is not None
    assert raw.parent.span_id == outer.context.span_id
    # The overlapping span itself still parents exactly (via ids).
    assert overlapping.parent is not None
    assert overlapping.parent.span_id == outer.context.span_id


async def test_our_root_nests_under_raw_otel_span(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, provider = otel_env
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("raw"):
        async with ai.telemetry.span("inner"):
            pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    inner, raw = spans["inner"], spans["raw"]
    assert inner.parent is not None
    assert inner.parent.span_id == raw.context.span_id
