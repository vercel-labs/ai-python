"""OpenTelemetry adapter: forwards spans to an otel tracer.

::

    from ai.telemetry import otel
    otel.install()  # uses the global TracerProvider

Span names and attributes follow the ``gen_ai`` semantic conventions,
so LLM-aware viewers (Phoenix, Braintrust, Langfuse, Datadog, ...)
render them natively.

The adapter also attaches each current-setting otel span to the otel
context in the task that opened it, so raw otel spans a user creates
inside a tool still parent correctly under ours — and our root spans
nest under any raw otel span the caller already has open.  Spans
opened with ``set_as_current=False`` are never attached: they don't
parent concurrent work in our tree, so they must not do it in otel's.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from typing import TYPE_CHECKING, Any

from .. import errors, telemetry

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace
except ModuleNotFoundError as exc:  # pragma: no cover
    raise errors.InstallationError(
        "could not import `opentelemetry`, which is required for the otel "
        'telemetry adapter, you can install it with `pip install "ai[otel]"` '
        'or `uv add "ai[otel]"`'
    ) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ..types import messages as messages_


def _messages_json(messages: list[messages_.Message]) -> str:
    return "[" + ",".join(m.model_dump_json() for m in messages) + "]"


def _name(sp: telemetry.Span) -> str:
    match sp.data:
        case telemetry.AiStreamSpanData() as d:
            return f"chat {d.model}"
        case telemetry.AiGenerateSpanData() as d:
            return f"generate_content {d.model}"
        case telemetry.ToolExecutionSpanData() as d:
            return f"execute_tool {d.tool_name}"
        case telemetry.RunSpanData() as d:
            return f"invoke_agent {d.agent}"
        case _:
            return sp.name


def _attributes(sp: telemetry.Span) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if sp.replay:
        attrs["ai.replay"] = True
    match sp.data:
        case telemetry.AiStreamSpanData() as d:
            attrs["gen_ai.operation.name"] = "chat"
            attrs["gen_ai.request.model"] = d.model
            attrs["gen_ai.input.messages"] = _messages_json(d.messages)
            if d.message is not None:
                attrs["gen_ai.output.messages"] = _messages_json([d.message])
            if d.usage is not None:
                attrs["gen_ai.usage.input_tokens"] = d.usage.input_tokens
                attrs["gen_ai.usage.output_tokens"] = d.usage.output_tokens
        case telemetry.AiGenerateSpanData() as d:
            attrs["gen_ai.operation.name"] = "generate_content"
            attrs["gen_ai.request.model"] = d.model
            if d.message is not None:
                attrs["gen_ai.output.messages"] = _messages_json([d.message])
        case telemetry.ToolExecutionSpanData() as d:
            attrs["gen_ai.operation.name"] = "execute_tool"
            attrs["gen_ai.tool.name"] = d.tool_name
            attrs["gen_ai.tool.call.id"] = d.tool_call_id
            if d.args is not None:
                attrs["gen_ai.tool.call.arguments"] = json.dumps(
                    d.args, default=str
                )
            if d.result is not None:
                attrs["gen_ai.tool.call.result"] = json.dumps(
                    d.result, default=str
                )
            if d.is_error:
                attrs["ai.tool.is_error"] = True
        case telemetry.RunSpanData() as d:
            attrs["gen_ai.operation.name"] = "invoke_agent"
            attrs["gen_ai.agent.name"] = d.agent
            attrs["gen_ai.request.model"] = d.model
        case telemetry.HookSpanData() as d:
            attrs["ai.hook.label"] = d.label
            attrs["ai.hook.type"] = d.hook_type
            attrs["ai.hook.status"] = d.status
        case telemetry.CustomSpanData() as d:
            for key, value in d.attributes.items():
                attrs[key] = (
                    value
                    if isinstance(value, str | bool | int | float)
                    else repr(value)
                )
    return attrs


# Our span/trace ids for the start_span call in flight, as ints.  Set
# around the (synchronous) call, read by _IdGenerator below.
_forced_ids: contextvars.ContextVar[tuple[int | None, int | None] | None] = (
    contextvars.ContextVar("otel_forced_ids", default=None)
)


class _IdGenerator:
    """Hands the SDK our span/trace ids; delegates otherwise.

    Installed over the provider's own generator so raw otel spans and
    other instrumentation are untouched — the override only applies
    while our adapter's ``start_span`` call is in flight.
    """

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback

    def __getattr__(self, name: str) -> Any:
        # The SDK's generator interface grows (is_trace_id_random, ...);
        # everything we don't override answers as the wrapped generator.
        return getattr(self._fallback, name)

    def generate_span_id(self) -> int:
        forced = _forced_ids.get()
        if forced is not None and forced[1] is not None:
            return forced[1]
        return self._fallback.generate_span_id()  # type: ignore[no-any-return]

    def generate_trace_id(self) -> int:
        forced = _forced_ids.get()
        if forced is not None and forced[0] is not None:
            return forced[0]
        return self._fallback.generate_trace_id()  # type: ignore[no-any-return]


def _hex_id(id_: str | None, bits: int) -> int | None:
    """``span_<hex>``/``trace_<hex>`` as an int; None if not otel-able.

    0 is otel's invalid id; the ``bits`` cap guards user-supplied
    ``SpanRef`` ids that don't fit otel's fixed widths.
    """
    if id_ is None:
        return None
    try:
        value = int(id_.rsplit("_", 1)[-1], 16)
    except ValueError:
        return None
    return value if 0 < value < 2**bits else None


def _remote_parent(sp: telemetry.Span) -> otel_trace.Span | None:
    """Build a stand-in for a parent with no live otel span here.

    Covers a ``SpanRef`` parent and a parent from a previous process
    life; the ids are reproducible, so they line up in the backend.
    """
    trace_id = _hex_id(sp.trace_id, 128)
    parent_id = _hex_id(sp.parent_id, 64)
    if trace_id is None or parent_id is None:
        return None
    return otel_trace.NonRecordingSpan(
        otel_trace.SpanContext(
            trace_id=trace_id,
            span_id=parent_id,
            is_remote=True,
            trace_flags=otel_trace.TraceFlags(otel_trace.TraceFlags.SAMPLED),
        )
    )


class _OtelAdapter(telemetry.Adapter):
    """One otel span per span frame, plus ``flush`` on the provider."""

    def __init__(
        self,
        tracer: otel_trace.Tracer,
        provider: otel_trace.TracerProvider,
        *,
        export_replays: bool,
    ) -> None:
        self._tracer = tracer
        self._provider = provider
        self._export_replays = export_replays
        # Our span id -> its live otel span, for parenting children.
        self._live: dict[str, otel_trace.Span] = {}

    async def wrap_span(
        self, span: telemetry.Span
    ) -> AsyncGenerator[None, Any]:
        if span.replay and not self._export_replays:
            return
        context = None
        if span.parent_id is not None:
            parent = self._live.get(span.parent_id) or _remote_parent(span)
            if parent is not None:
                context = otel_trace.set_span_in_context(parent)
        forced = _forced_ids.set(
            (_hex_id(span.trace_id, 128), _hex_id(span.id, 64))
        )
        try:
            otel_span = self._tracer.start_span(
                _name(span), context=context, start_time=span.started_at
            )
        finally:
            _forced_ids.reset(forced)
        self._live[span.id] = otel_span
        token = (
            otel_context.attach(otel_trace.set_span_in_context(otel_span))
            if span.set_as_current
            else None
        )
        try:
            # span end resumes with None
            while (ev := (yield)) is not None:
                otel_span.add_event(
                    ev.name,
                    # otel attribute values must be str | bool | int | float
                    {
                        k: v
                        if isinstance(v, str | bool | int | float)
                        else repr(v)
                        for k, v in ev.attributes.items()
                    },
                    timestamp=ev.time_ns,
                )
        finally:
            self._live.pop(span.id, None)
            if token is not None:
                otel_context.detach(token)
            for key, value in _attributes(span).items():
                otel_span.set_attribute(key, value)
            if span.error is not None:
                if isinstance(span.error, Exception):
                    otel_span.record_exception(span.error)
                otel_span.set_status(
                    otel_trace.StatusCode.ERROR, str(span.error)
                )
            otel_span.end(end_time=span.ended_at)

    async def flush(self) -> None:
        force_flush = getattr(self._provider, "force_flush", None)
        if force_flush is not None:
            # Blocks until exporters drain; keep the loop free.
            await asyncio.to_thread(force_flush)


def install(
    *,
    tracer_provider: otel_trace.TracerProvider | None = None,
    export_replays: bool = False,
) -> telemetry.Adapter:
    """Create the otel adapter, register it, and return it.

    Uses the global tracer provider unless one is passed.

    ``export_replays=False`` (the default) skips spans marked
    ``replay=True``: under deterministic replay every span was live
    exactly once, so live-only export is complete and duplicate-free.
    Pass ``True`` for backends that overwrite spans by id — replay
    emissions can carry fields that only settle after a resume.
    """
    provider = (
        tracer_provider
        if tracer_provider is not None
        else otel_trace.get_tracer_provider()
    )
    # Carry our ids through as the otel ids where the provider allows
    # it (the SDK provider does).  Cross-process parenting and replay
    # dedup both key on ids being reproducible in the backend.
    fallback = getattr(provider, "id_generator", None)
    if fallback is not None and not isinstance(fallback, _IdGenerator):
        provider.id_generator = (  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
            _IdGenerator(fallback)
        )
    tracer = otel_trace.get_tracer("ai", tracer_provider=provider)
    adapter = _OtelAdapter(tracer, provider, export_replays=export_replays)
    telemetry.register(adapter)
    return adapter
