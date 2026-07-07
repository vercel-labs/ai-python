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
        case telemetry.LoopTurnSpanData() as d:
            return f"loop_turn {d.index}"
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
        case telemetry.LoopTurnSpanData() as d:
            attrs["ai.loop_turn.index"] = d.index
        case telemetry.CustomSpanData() as d:
            for key, value in d.attributes.items():
                attrs[key] = (
                    value
                    if isinstance(value, str | bool | int | float)
                    else repr(value)
                )
    return attrs


def install(
    *, tracer_provider: otel_trace.TracerProvider | None = None
) -> telemetry.WrapSpanAdapter:
    """Create the otel adapter, register it, and return it.

    Uses the global tracer provider unless one is passed.
    """
    tracer = otel_trace.get_tracer("ai", tracer_provider=tracer_provider)

    @telemetry.wrap_span
    async def otel_spans(span: telemetry.Span) -> AsyncGenerator[None]:
        # Parenting is ambient: the current otel context holds the
        # parent (ours attach below; a raw otel span the caller has
        # open works the same way), and it mirrors the framework's
        # parenting because only current-setting spans attach.
        otel_span = tracer.start_span(_name(span), start_time=span.started_at)
        # Mirror the framework's currentness: a set_as_current=False
        # span doesn't parent concurrent work in our tree, so it must
        # not become current in the otel context either.
        token = (
            otel_context.attach(otel_trace.set_span_in_context(otel_span))
            if span.set_as_current
            else None
        )
        try:
            yield
        finally:
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

    telemetry.register(otel_spans)
    return otel_spans
