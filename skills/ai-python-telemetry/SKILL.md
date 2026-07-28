---
name: ai-python-telemetry
description: Use when adding telemetry, tracing, or OpenTelemetry export to AI SDK for Python, or when reporting spans from durable or serverless execution.
metadata:
  sdk-version: "0.4.0"
---

# ai-python-telemetry

<!-- Outline only. New in 0.4.0. Prose and code examples to be written. -->

- `ai.experimental_telemetry` is experimental: it may change or be removed.

## Enable telemetry

- Nothing is reported until an adapter is registered or a sink is routed;
  with telemetry off, spans are no-ops (`id=""`, no clock reads, `push()`
  does nothing). Check with `is_enabled()`.
- OpenTelemetry export: perform the standard OpenTelemetry ceremony (tracer
  provider + OTLP exporter), then
  `ai.experimental_telemetry.register(otel.OtelAdapter())`. Requires the
  `ai[otel]` extra.
- Message content capture is opt-in: `otel.OtelAdapter(capture_content=True)`
  or `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`.
- Call `OtelAdapter.flush()` before a checkpoint and `OtelAdapter.shutdown()`
  at process exit; the exporter buffers.

## Framework spans

- The framework opens spans automatically: `RunSpanData`, `LoopTurnSpanData`,
  `AiStreamSpanData`, `AiGenerateSpanData`, `ToolExecutionSpanData`,
  `HookSpanData`.
- Dispatch on the type of `span.data` in adapters; span events include
  `FIRST_TOKEN`, `RESPONSE_COMPLETE`, `HOOK_DEFERRED`, `HOOK_RESOLVED`,
  `HOOK_CANCELLED`.

## User spans

- `async with span("name") as sp:` — stamps, pushes, nests, and
  records exceptions automatically.
- `sp.set_attrs(...)` for attributes; `sp.add_event(name)` for milestones.
- Typed spans: any Pydantic model with a `kind` field is `SpanData`; open
  with `span(MyData(...))` and assign `sp.data` fields directly.

## Custom adapters

- Protocol: `on_span_start` / `on_span_event` / `on_span_end` (all required),
  registered with `register()` — it raises `TypeError` otherwise. Failures
  are logged and skipped.
- Prefer the `@adapter` decorator for vendor bridges: one async generator per
  span, yield loop until `None`, then read `span.data` / `span.error`. Also
  works on a class whose `__call__` is an async generator method, for
  adapters with configuration or state.
- Subclass `OtelAdapter` and override `span_name()` / `span_attrs()`
  to adjust OpenTelemetry output.

## Durable and serverless execution

- Inside a workflow body, route spans to a sink instead of the adapters and
  keep delivery in steps; with deterministic clock and random installed,
  agent runs and the `span()` context manager work in the body. Use the
  low-level API for spans that cross process boundaries.
- Collect in the body: `sink = DictSink()`, run under
  `async with use_sink(sink):`, then serialize `sink.finished_spans` with
  `model_dump(mode="json")`.
- Deliver from a step: `await push_all(payload)`.
- Manual lifecycle: `create_span()` then `stamp_start()` / `stamp_end()` /
  `push()`; nothing is reported except by pushing.
- Continue a trace across processes: restore with `Span.model_validate(...)`,
  parent under it with `async with use_span(restored):` or
  `span(..., parent=restored)`.
- Deterministic timestamps: `use_time(workflow.time_ns)`; read the ambient
  clock with `now_ns()`.
- `use_sink`, `use_span`, and `use_time` are async context managers; they also
  work as decorators on async functions, but never with a plain `with`.
- Errors serialize as `SpanError` (`SpanError.from_exception(exc)`), so spans
  that failed in another process still report.
