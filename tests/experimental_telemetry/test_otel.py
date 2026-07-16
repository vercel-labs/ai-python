"""Otel adapter: names, attributes, parenting, context bridging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import ai
from ai.experimental_telemetry import otel

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
    ai.experimental_telemetry.unregister(adapter)


async def test_nesting_names_and_attributes(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    async with ai.experimental_telemetry.span("outer", foo="bar"):
        async with ai.experimental_telemetry.span("inner"):
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
        async with ai.experimental_telemetry.span("failing"):
            raise ValueError("boom")

    (span,) = exporter.get_finished_spans()
    assert not span.status.is_ok
    # The error arrives as serializable data (never a live exception —
    # the span may have failed in another process), so it exports as
    # the status description.
    assert span.status.description == "ValueError: boom"


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
    # records it: ERROR status carrying the failure's type and message.
    assert not span.status.is_ok
    assert span.status.description == "ValueError: nope"


async def test_span_events_exported_with_original_timestamps(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env

    class Marker:
        # A stable repr: pushes snapshot the span, so the adapter sees
        # a copy of this value, not the original object.
        def __repr__(self) -> str:
            return "<marker>"

    marker = Marker()
    async with ai.experimental_telemetry.span("s") as sp:
        first = ai.experimental_telemetry.SpanEvent(
            name="first_token",
            time_ns=ai.experimental_telemetry.now_ns(),
            attributes={"event_type": "TextStart"},
        )
        sp.events.append(first)
        await sp.push()
        second = ai.experimental_telemetry.SpanEvent(
            name="custom",
            time_ns=ai.experimental_telemetry.now_ns(),
            attributes={"obj": marker},
        )
        sp.events.append(second)
        await sp.push()

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


async def test_raw_otel_span_nests_under_ours(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, provider = otel_env
    tracer = provider.get_tracer("test")
    async with ai.experimental_telemetry.span("outer"):
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
    async with ai.experimental_telemetry.span("outer"):
        async with ai.experimental_telemetry.span(
            "overlapping", set_as_current=False
        ):
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
        async with ai.experimental_telemetry.span("inner"):
            pass

    spans = {s.name: s for s in exporter.get_finished_spans()}
    inner, raw = spans["inner"], spans["raw"]
    assert inner.parent is not None
    assert inner.parent.span_id == raw.context.span_id


async def test_span_finished_elsewhere_exports_and_parents(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    # "Process one": mint the span and carry it as data.  No push —
    # the record is delivered whole once the work completes.
    turn = ai.experimental_telemetry.create_span("turn")
    turn.started_at = ai.experimental_telemetry.now_ns()
    payload = turn.model_dump(mode="json")

    # "Process two": live children under the restored span.
    restored = ai.experimental_telemetry.Span.model_validate(payload)
    with ai.experimental_telemetry.use_span(restored):
        async with ai.experimental_telemetry.span("child"):
            pass

    # "Process three": finish the span and push the complete record.
    done = ai.experimental_telemetry.Span.model_validate(payload)
    done.ended_at = ai.experimental_telemetry.now_ns()
    done.error = ai.experimental_telemetry.SpanError(
        type="TurnError", message="nope"
    )
    await done.push()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    child, turn_span = spans["child"], spans["turn"]
    # The child parented on ids derived from the framework's before the
    # turn span existed anywhere in otel; the finished record exports
    # under those exact ids, so the tree lines up.
    assert child.parent is not None
    assert child.parent.span_id == turn_span.context.span_id
    assert child.context.trace_id == turn_span.context.trace_id
    # Timestamps and outcome come from the record, not the clock here.
    assert turn_span.start_time == turn.started_at
    assert turn_span.end_time == done.ended_at
    assert not turn_span.status.is_ok
    assert turn_span.status.description == "TurnError: nope"


async def test_otel_ids_derived_from_framework_ids(
    otel_env: tuple[InMemorySpanExporter, TracerProvider],
) -> None:
    exporter, _ = otel_env
    async with ai.experimental_telemetry.span("s") as sp:
        pass

    (span,) = exporter.get_finished_spans()
    # Deterministic identity: replays and cross-process re-emission of
    # the same framework span come out under the same otel ids.
    assert span.context.span_id == otel._derive_span_id(sp.id)
    assert span.context.trace_id == otel._derive_trace_id(sp.trace_id)


async def test_subclass_can_enrich_names_and_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    class Enriched(otel.OtelAdapter):
        def span_name(self, span_: ai.experimental_telemetry.Span, /) -> str:
            return f"seal:{super().span_name(span_)}"

        def span_attributes(
            self, span_: ai.experimental_telemetry.Span, /
        ) -> dict[str, Any]:
            return super().span_attributes(span_) | {"extra": True}

    adapter = Enriched(tracer_provider=provider)
    ai.experimental_telemetry.register(adapter)
    try:
        async with ai.experimental_telemetry.span("s", foo="bar"):
            pass
    finally:
        ai.experimental_telemetry.unregister(adapter)

    (span,) = exporter.get_finished_spans()
    assert span.name == "seal:s"
    assert span.attributes is not None
    assert span.attributes["foo"] == "bar"
    assert span.attributes["extra"] is True
