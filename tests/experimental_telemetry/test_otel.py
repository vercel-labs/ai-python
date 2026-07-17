"""Otel adapter: names, attributes, parenting, context bridging."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind

import ai
from ai.experimental_telemetry import otel
from ai.models.core import params as core_params
from ai.types import messages as messages_
from ai.types import usage as usage_

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def otel_env() -> Iterator[tuple[InMemorySpanExporter, TracerProvider]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = otel.install(tracer_provider=provider, capture_content=True)
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
    # Inference spans are CLIENT per semconv.
    assert span.kind is SpanKind.CLIENT
    attrs = span.attributes
    assert attrs is not None
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "mock"
    assert attrs["gen_ai.request.model"] == "mock-model"
    assert attrs["gen_ai.request.stream"] is True
    ttfc = attrs["gen_ai.response.time_to_first_chunk"]
    assert isinstance(ttfc, float) and ttfc >= 0
    # Content in the semconv message shape, not our native model dump.
    assert json.loads(str(attrs["gen_ai.input.messages"])) == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert json.loads(str(attrs["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "hello"}],
            "finish_reason": "stop",
        }
    ]


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
    # records it: ERROR status carrying the failure's type and message,
    # plus the semconv ``error.type`` attribute.
    assert not span.status.is_ok
    assert span.status.description == "ValueError: nope"
    assert span.attributes is not None
    assert span.attributes["error.type"] == "ValueError"
    assert span.attributes["gen_ai.tool.type"] == "function"


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


async def test_content_capture_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(otel.CAPTURE_CONTENT_ENV, raising=False)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = otel.install(tracer_provider=provider)
    try:
        mock_llm([[text_msg("hello")]])
        async with ai.stream(MOCK_MODEL, [ai.user_message("hi")]) as stream:
            async for _ in stream:
                pass
    finally:
        ai.experimental_telemetry.unregister(adapter)

    (span,) = exporter.get_finished_spans()
    attrs = span.attributes
    assert attrs is not None
    # Telemetry attributes still export; content is Opt-In per semconv.
    assert attrs["gen_ai.request.model"] == "mock-model"
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert "gen_ai.system_instructions" not in attrs
    assert "gen_ai.tool.definitions" not in attrs


async def test_content_capture_env_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(otel.CAPTURE_CONTENT_ENV, "span_only")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    adapter = otel.install(tracer_provider=provider)
    try:
        mock_llm([[text_msg("hello")]])
        async with ai.stream(MOCK_MODEL, [ai.user_message("hi")]) as stream:
            async for _ in stream:
                pass
    finally:
        ai.experimental_telemetry.unregister(adapter)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert "gen_ai.input.messages" in span.attributes


def _stream_span(
    **data_kwargs: Any,
) -> ai.experimental_telemetry.Span[Any]:
    data_kwargs.setdefault("messages", [ai.user_message("hi")])
    data = ai.experimental_telemetry.AiStreamSpanData(
        model="mock-model", **data_kwargs
    )
    return ai.experimental_telemetry.Span(
        name="ai_stream", data=data, id="span-1", trace_id="trace-1"
    )


def test_request_and_usage_attributes() -> None:
    params = core_params.InferenceRequestParams(
        sampling={
            core_params.TemperatureSamplerParams: (
                core_params.TemperatureSamplerParams(temperature=0.5)
            ),
            core_params.TopPSamplerParams: (
                core_params.TopPSamplerParams(top_p=0.9)
            ),
            # Left at the provider-default sentinel: must not export.
            core_params.TopKSamplerParams: core_params.TopKSamplerParams(),
        },
        reasoning=core_params.ReasoningParams(effort="high"),
        output=core_params.OutputParams(max_tokens=128),
    )
    usage = usage_.Usage(
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=2,
        cache_read_tokens=3,
        cache_write_tokens=4,
    )
    sp = _stream_span(params=params, usage=usage, provider="mock")

    attrs = otel._attributes(sp, capture_content=False)
    assert attrs["gen_ai.provider.name"] == "mock"
    assert attrs["gen_ai.request.temperature"] == 0.5
    assert attrs["gen_ai.request.top_p"] == 0.9
    assert "gen_ai.request.top_k" not in attrs
    assert attrs["gen_ai.request.max_tokens"] == 128
    assert attrs["gen_ai.request.reasoning.level"] == "high"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 2
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 3
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 4
    assert "gen_ai.input.messages" not in attrs


def test_request_attributes_from_json_restored_params() -> None:
    # A span restored from JSON carries params as plain dicts.
    sp = _stream_span(
        params={
            "sampling": {"temperature": {"temperature": 0.25}},
            "output": {"max_tokens": 64},
            "reasoning": {"effort": "low"},
        }
    )
    attrs = otel._attributes(sp, capture_content=False)
    assert attrs["gen_ai.request.temperature"] == 0.25
    assert attrs["gen_ai.request.max_tokens"] == 64
    assert attrs["gen_ai.request.reasoning.level"] == "low"


def test_response_identity_attributes() -> None:
    sp = _stream_span(
        output_type="MyModel",
        message=text_msg("truncated"),
        finish_reason="length",
        response_id="resp-1",
        response_model="mock-model-v2",
    )
    attrs = otel._attributes(sp, capture_content=True)
    assert attrs["gen_ai.response.finish_reasons"] == ["length"]
    assert attrs["gen_ai.response.id"] == "resp-1"
    assert attrs["gen_ai.response.model"] == "mock-model-v2"
    # Structured output requested: the semconv output kind, not the
    # Python type name.
    assert attrs["gen_ai.output.type"] == "json"
    # The captured finish reason wins over the inferred fallback.
    (out,) = json.loads(attrs["gen_ai.output.messages"])
    assert out["finish_reason"] == "length"


def test_tool_span_description() -> None:
    data = ai.experimental_telemetry.ToolExecutionSpanData(
        tool_name="lookup",
        tool_call_id="tc-1",
        tool_description="Look things up.",
    )
    sp = ai.experimental_telemetry.Span(
        name="tool_execution", data=data, id="span-1", trace_id="trace-1"
    )
    attrs = otel._attributes(sp, capture_content=False)
    assert attrs["gen_ai.tool.description"] == "Look things up."


def test_semconv_message_content_shapes() -> None:
    messages = [
        messages_.Message(
            role="system", parts=[messages_.TextPart(text="be nice")]
        ),
        messages_.Message(
            role="user",
            parts=[
                messages_.TextPart(text="hi"),
                messages_.FilePart(
                    data="https://e.com/a.png", media_type="image/png"
                ),
                messages_.FilePart(data="QUJD", media_type="application/pdf"),
            ],
        ),
        messages_.Message(
            role="assistant",
            parts=[
                messages_.ReasoningPart(text="hmm"),
                messages_.BuiltinToolCallPart(
                    tool_call_id="b1",
                    tool_name="web_search",
                    tool_args='{"q": "x"}',
                ),
                messages_.BuiltinToolReturnPart(
                    tool_call_id="b1",
                    tool_name="web_search",
                    result={"hits": 1},
                ),
            ],
        ),
        messages_.Message(
            role="tool",
            parts=[
                messages_.ToolResultPart(
                    tool_call_id="t1", tool_name="lookup", result={"ok": True}
                )
            ],
        ),
    ]
    sp = _stream_span(messages=messages, message=tool_call_msg(name="lookup"))

    attrs = otel._attributes(sp, capture_content=True)
    # System messages move out of input.messages, as a flat parts list.
    assert json.loads(attrs["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "be nice"}
    ]
    inputs = json.loads(attrs["gen_ai.input.messages"])
    assert [m["role"] for m in inputs] == ["user", "assistant", "tool"]
    assert inputs[0]["parts"] == [
        {"type": "text", "content": "hi"},
        {
            "type": "uri",
            "modality": "image",
            "mime_type": "image/png",
            "uri": "https://e.com/a.png",
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "application/pdf",
            "content": "QUJD",
        },
    ]
    assert inputs[1]["parts"] == [
        {"type": "reasoning", "content": "hmm"},
        {
            "type": "server_tool_call",
            "id": "b1",
            "name": "web_search",
            "server_tool_call": {
                "type": "web_search",
                "arguments": {"q": "x"},
            },
        },
        {
            "type": "server_tool_call_response",
            "id": "b1",
            "server_tool_call_response": {
                "type": "web_search",
                "response": {"hits": 1},
            },
        },
    ]
    assert inputs[2]["parts"] == [
        {"type": "tool_call_response", "id": "t1", "response": {"ok": True}}
    ]
    (out,) = json.loads(attrs["gen_ai.output.messages"])
    assert out["finish_reason"] == "tool_call"
    assert out["parts"] == [
        {"type": "tool_call", "id": "tc-1", "name": "lookup", "arguments": {}}
    ]


def test_generate_span_attributes() -> None:
    data = ai.experimental_telemetry.AiGenerateSpanData(
        model="mock-model",
        messages=[ai.user_message("draw")],
        params=core_params.ImageParams(n=2, seed=7),
        provider="mock",
    )
    sp = ai.experimental_telemetry.Span(
        name="ai_generate", data=data, id="span-1", trace_id="trace-1"
    )
    attrs = otel._attributes(sp, capture_content=False)
    assert attrs["gen_ai.operation.name"] == "generate_content"
    assert attrs["gen_ai.provider.name"] == "mock"
    assert attrs["gen_ai.request.choice.count"] == 2
    assert attrs["gen_ai.request.seed"] == 7
    assert attrs["gen_ai.output.type"] == "image"

    data = ai.experimental_telemetry.AiGenerateSpanData(
        model="mock-model",
        messages=[ai.user_message("film")],
        params=core_params.VideoParams(),
        provider="mock",
    )
    sp = ai.experimental_telemetry.Span(
        name="ai_generate", data=data, id="span-2", trace_id="trace-1"
    )
    attrs = otel._attributes(sp, capture_content=False)
    assert attrs["gen_ai.output.type"] == "video"
    assert "gen_ai.request.choice.count" not in attrs


def test_run_span_attributes() -> None:
    data = ai.experimental_telemetry.RunSpanData(
        agent="MyAgent",
        model="anthropic/claude-x",
        messages=[ai.user_message("hi")],
        provider="ai-gateway",
        tool_names=["lookup"],
        output_type="MyModel",
        usage=usage_.Usage(input_tokens=30, output_tokens=5),
    )
    sp = ai.experimental_telemetry.Span(
        name="run", data=data, id="span-1", trace_id="trace-1"
    )
    attrs = otel._attributes(sp, capture_content=True)
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.name"] == "MyAgent"
    # Gateway model ids carry the actual provider as their prefix.
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "anthropic/claude-x"
    # Structured output requested: the semconv output kind, not the
    # Python type name.
    assert attrs["gen_ai.output.type"] == "json"
    assert attrs["gen_ai.usage.input_tokens"] == 30
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert json.loads(attrs["gen_ai.tool.definitions"]) == [
        {"type": "function", "name": "lookup"}
    ]
    assert "gen_ai.input.messages" in attrs
