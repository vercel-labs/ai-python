"""OpenTelemetry adapter: forwards spans to an otel tracer.

Experimental: not part of the stable API, may change or be removed.

::

    from ai.experimental_telemetry import otel
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

Identity: otel ids are derived deterministically from the framework's
span/trace ids (a truncated hash), via an id generator installed on the
SDK tracer provider.  That is what makes spans that cross processes
line up: a span finished and pushed in a different process than it
started in exports under the same otel identity its children already
parented to, and a re-emitted span comes out under the same ids instead
of duplicating the tree.  A span whose parent is not live in this
process is parented through those derived ids directly.

With a non-SDK tracer provider (no id generator to install) the
adapter still works for in-process traces, but ids fall back to
whatever the tracer mints, so cross-process parenting degrades; a
warning is logged once if that comes up.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from .. import errors
from .. import experimental_telemetry as telemetry

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk import trace as sdk_trace
    from opentelemetry.sdk.trace import id_generator as sdk_id_generator
except ModuleNotFoundError as exc:  # pragma: no cover
    raise errors.InstallationError(
        "could not import `opentelemetry`, which is required for the otel "
        'telemetry adapter, you can install it with `pip install "ai[otel]"` '
        'or `uv add "ai[otel]"`'
    ) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ..types import messages as messages_

logger = logging.getLogger(__name__)


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


def _derive_trace_id(id_: str) -> int:
    """Derive a stable, nonzero otel trace id (128 bits) from one of ours."""
    return int.from_bytes(hashlib.sha256(id_.encode()).digest()[:16]) or 1


def _derive_span_id(id_: str) -> int:
    """Derive a stable, nonzero otel span id (64 bits) from one of ours."""
    return int.from_bytes(hashlib.sha256(id_.encode()).digest()[:8]) or 1


# The otel Tracer offers no way to dictate a span's ids, but derived
# ids must come out exactly (see module docstring); the generator
# installed on the SDK provider honors a preset while it is set.
_preset_ids: contextvars.ContextVar[tuple[int, int] | None] = (
    contextvars.ContextVar("otel_preset_ids", default=None)
)


class _PresetIdGenerator(sdk_id_generator.RandomIdGenerator):
    """Uses the preset ids when set; defers to ``inner`` otherwise."""

    def __init__(self, inner: sdk_id_generator.IdGenerator) -> None:
        self._inner = inner

    def generate_trace_id(self) -> int:
        preset = _preset_ids.get()
        return (
            preset[0] if preset is not None else self._inner.generate_trace_id()
        )

    def generate_span_id(self) -> int:
        preset = _preset_ids.get()
        return (
            preset[1] if preset is not None else self._inner.generate_span_id()
        )


class OtelAdapter(telemetry.Adapter):
    """Maps framework spans onto otel spans; see the module docstring."""

    def __init__(
        self, *, tracer_provider: otel_trace.TracerProvider | None = None
    ) -> None:
        provider = tracer_provider or otel_trace.get_tracer_provider()
        self._provider = provider
        self._live: dict[str, otel_trace.Span] = {}
        self._deterministic_ids = False
        self._warned_random_ids = False
        if isinstance(provider, sdk_trace.TracerProvider):
            provider.id_generator = _PresetIdGenerator(provider.id_generator)
            self._deterministic_ids = True
        self._tracer = otel_trace.get_tracer("ai", tracer_provider=provider)

    def span_name(self, span_: telemetry.Span, /) -> str:
        """Return the exported otel span name.  Override to customize."""
        return _name(span_)

    def span_attributes(self, span_: telemetry.Span, /) -> dict[str, Any]:
        """Return the attributes set at span end.  Override to enrich.

        ::

            class MyAdapter(otel.OtelAdapter):
                def span_attributes(self, span_):
                    return super().span_attributes(span_) | {"k": "v"}
        """
        return _attributes(span_)

    def flush(self) -> None:
        """Flush the provider's exporters, when it has any (SDK provider)."""
        force_flush = getattr(self._provider, "force_flush", None)
        if force_flush is not None:
            force_flush()

    def shutdown(self) -> None:
        """Flush and stop the provider; spans pushed after this are lost."""
        shutdown = getattr(self._provider, "shutdown", None)
        if shutdown is not None:
            shutdown()

    def _parent_context(
        self, span_: telemetry.Span
    ) -> otel_context.Context | None:
        # ``None`` means the ambient otel context: the parent's otel
        # span is attached there when it is live in this process, and
        # for our roots it lets any raw otel span the caller holds
        # adopt the trace.
        if span_.parent_id is None or span_.parent_id in self._live:
            return None
        # The parent is not live here — it started in another process,
        # or this span arrived as a finished record.  Parent on the
        # derived identity in an empty context so the pieces line up
        # when the parent itself exports.
        if not self._deterministic_ids and not self._warned_random_ids:
            self._warned_random_ids = True
            logger.warning(
                "otel adapter: continuing a trace from another process "
                "needs an SDK tracer provider to control span ids; ids "
                "will not line up across processes"
            )
        return otel_trace.set_span_in_context(
            otel_trace.NonRecordingSpan(
                otel_trace.SpanContext(
                    trace_id=_derive_trace_id(span_.trace_id),
                    span_id=_derive_span_id(span_.parent_id),
                    is_remote=True,
                    trace_flags=otel_trace.TraceFlags(
                        otel_trace.TraceFlags.SAMPLED
                    ),
                )
            ),
            otel_context.Context(),
        )

    async def wrap_span(
        self, span_: telemetry.Span, /
    ) -> AsyncGenerator[None, Any]:
        parent_context = self._parent_context(span_)
        preset_token = _preset_ids.set(
            (_derive_trace_id(span_.trace_id), _derive_span_id(span_.id))
        )
        try:
            otel_span = self._tracer.start_span(
                self.span_name(span_),
                context=parent_context,
                start_time=span_.started_at,
            )
        finally:
            _preset_ids.reset(preset_token)
        self._live[span_.id] = otel_span
        token = (
            otel_context.attach(otel_trace.set_span_in_context(otel_span))
            if span_.set_as_current
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
            if token is not None:
                otel_context.detach(token)
            self._live.pop(span_.id, None)
            for key, value in self.span_attributes(span_).items():
                otel_span.set_attribute(key, value)
            if span_.error is not None:
                otel_span.set_status(
                    otel_trace.StatusCode.ERROR,
                    f"{span_.error.type}: {span_.error.message}",
                )
            otel_span.end(end_time=span_.ended_at)


def install(
    *, tracer_provider: otel_trace.TracerProvider | None = None
) -> OtelAdapter:
    """Create the otel adapter, register it, and return it.

    Uses the global tracer provider unless one is passed.
    """
    adapter = OtelAdapter(tracer_provider=tracer_provider)
    telemetry.register(adapter)
    return adapter
