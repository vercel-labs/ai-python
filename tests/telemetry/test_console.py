"""Console adapter: live lines and end-of-trace tree."""

from __future__ import annotations

import io

import ai
from ai.telemetry import console


async def test_console_prints_tree() -> None:
    out = io.StringIO()
    adapter = console.ConsoleAdapter(out=out)
    ai.telemetry.register(adapter)
    try:
        async with ai.telemetry.span("outer"):
            async with ai.telemetry.span("inner", k=1):
                pass
    finally:
        ai.telemetry.unregister(adapter)

    text = out.getvalue()
    assert "▸ outer" in text
    assert "▸   inner (k=1)" in text
    assert "└─ inner (k=1)" in text
    assert "trace " in text


async def test_console_prints_live_span_events() -> None:
    out = io.StringIO()
    adapter = console.ConsoleAdapter(out=out)
    ai.telemetry.register(adapter)
    try:
        async with ai.telemetry.span("outer") as sp:
            await sp.add_event("first_token", event_type="TextStart")
    finally:
        ai.telemetry.unregister(adapter)

    text = out.getvalue()
    assert "·   first_token +" in text
    assert "ms (event_type='TextStart')" in text


async def test_console_tree_uses_response_complete_duration() -> None:
    out = io.StringIO()
    adapter = console.ConsoleAdapter(out=out)
    span = ai.telemetry.Span(
        name="ai_stream",
        data=ai.telemetry.AiStreamSpanData(model="m", messages=[]),
        id="span-1",
        trace_id="trace-1",
        parent_id=None,
        started_at=0,
        ended_at=5_000_000_000,
        span_events=[
            ai.telemetry.SpanEvent(
                name=ai.telemetry.RESPONSE_COMPLETE,
                time_ns=1_500_000_000,
                attributes={},
            )
        ],
    )
    adapter.on_span_start(span)
    adapter.on_span_end(span)

    text = out.getvalue()
    # The reported duration is the model latency, not the span
    # lifetime (which includes tool dispatch while the stream is open).
    assert "1.50s" in text
    assert "5.00s" not in text
