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
  does nothing). Check with `enabled()`.
- OpenTelemetry export: `from ai.experimental_telemetry import otel;
  otel.install()`. Requires the `ai[otel]` extra.
- Message content capture is opt-in: `otel.install(capture_content=True)` or
  `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`.
- Call `flush()` before process exit or a checkpoint; vendor SDKs buffer.

## Framework spans

- The framework opens spans automatically: `RunSpanData`, `LoopTurnSpanData`,
  `AiStreamSpanData`, `AiGenerateSpanData`, `ToolExecutionSpanData`,
  `HookSpanData`.
- Dispatch on the type of `span.data` in adapters; span events include
  `FIRST_TOKEN`, `RESPONSE_COMPLETE`, `HOOK_DEFERRED`, `HOOK_RESOLVED`,
  `HOOK_CANCELLED`.

## User spans

- `async with span("name", key=value) as sp:` — stamps, pushes, nests, and
  records exceptions automatically.
- `sp.set(...)` for attributes; `sp.add_event(name)` for milestones.
- Typed spans: any Pydantic model with a `kind` field is `SpanData`; open
  with `span(MyData(...))` and assign `sp.data` fields directly.

## Custom adapters

- Protocol: optional `on_span_start` / `on_span_event` / `on_span_end`,
  registered with `register()`. Failures are logged and skipped.
- Prefer `wrap_span` for vendor bridges: one async generator per span,
  yield loop until `None`, then read `span.data` / `span.error`.
- Subclass `OtelAdapter` and override `span_name()` / `span_attributes()`
  to adjust OpenTelemetry output.

## Durable and serverless execution

- Do not use the `span()` context manager inside a workflow body; use the
  low-level API and keep delivery in steps.
- Collect in the body: `collector = Collector()`, run under
  `with use_sink(collector):`, then serialize `collector.finished` with
  `model_dump(mode="json")`.
- Deliver from a step: `await push_all(payload)`.
- Manual lifecycle: `create_span()` then `stamp_start()` / `stamp_end()` /
  `push()`; nothing is reported except by pushing.
- Continue a trace across processes: restore with `Span.model_validate(...)`,
  parent under it with `use_span(restored)` or `span(..., parent=restored)`.
- Deterministic timestamps: `use_clock(workflow.time_ns)`; read the ambient
  clock with `now_ns()`.
- Errors serialize as `SpanError` (`SpanError.from_exception(exc)`), so spans
  that failed in another process still report.
