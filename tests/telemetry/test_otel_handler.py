"""OTel handler: span names, hierarchy, and attributes."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import ai
from ai.telemetry.otel import OtelHandler

from ..conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg


@pytest.fixture
def spans() -> Generator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ai.telemetry.enable(OtelHandler(tracer_provider=provider))
    yield exporter
    ai.telemetry.disable()
    provider.shutdown()


@ai.tool
async def double(x: int) -> int:
    """Double a number."""
    return x * 2


async def test_text_only_spans(spans: InMemorySpanExporter) -> None:
    """Text-only run produces ai.run > ai.stream span hierarchy."""
    my_agent = ai.agent()

    mock_llm([[text_msg("Hello!")]])
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Hi")]):
        pass

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert "ai.run" in names
    assert "ai.stream" in names

    run_span = next(s for s in finished if s.name == "ai.run")
    stream_span = next(s for s in finished if s.name == "ai.stream")

    # ai.stream is a child of ai.run
    stream_parent = stream_span.parent
    assert stream_parent is not None
    run_ctx = run_span.context
    assert run_ctx is not None
    assert stream_parent.span_id == run_ctx.span_id

    # run_id attribute is set
    assert run_span.attributes is not None
    assert run_span.attributes.get("ai.run.id") != ""


async def test_tool_call_spans(spans: InMemorySpanExporter) -> None:
    """Tool-calling run produces ai.tool spans with correct attributes."""
    my_agent = ai.agent(tools=[double])

    mock_llm(
        [
            [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 5}')],
            [text_msg("10")],
        ]
    )
    async for _m in my_agent.run(MOCK_MODEL, [ai.user_message("Double 5")]):
        pass

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert names.count("ai.stream") == 2
    assert "ai.tool" in names

    tool_span = next(s for s in finished if s.name == "ai.tool")
    assert tool_span.attributes is not None
    assert tool_span.attributes.get("gen_ai.tool.name") == "double"
    assert tool_span.attributes.get("gen_ai.tool.call.id") == "tc-1"

    # ai.tool is a child of ai.run (tools execute between steps)
    run_span = next(s for s in finished if s.name == "ai.run")
    tool_parent = tool_span.parent
    assert tool_parent is not None
    run_ctx = run_span.context
    assert run_ctx is not None
    assert tool_parent.span_id == run_ctx.span_id
