"""Otel adapter: names, attributes, parenting, context bridging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import ai
from ai.telemetry import otel

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
