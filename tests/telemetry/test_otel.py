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

from ..conftest import MOCK_MODEL, mock_llm, text_msg

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
