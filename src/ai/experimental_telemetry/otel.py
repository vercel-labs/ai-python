"""OpenTelemetry adapter.

Experimental: not part of the stable API, may change or be removed.

::

    from ai.experimental_telemetry import otel
    otel.install()  # uses the global TracerProvider

Follows the ``gen_ai`` semantic conventions.

There's a certain amount of jank in this adapter, because we want it to work in
the durable execution case, which conflicts with the OTel spec.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pydantic

from .. import errors
from .. import experimental_telemetry as telemetry
from ..types import messages as messages_

try:
    import opentelemetry.context
    import opentelemetry.sdk.trace.id_generator
    import opentelemetry.trace
except ModuleNotFoundError as exc:  # pragma: no cover
    raise errors.InstallationError(
        "could not import `opentelemetry`, which is required for the otel "
        'telemetry adapter, you can install it with `pip install "ai[otel]"` '
        'or `uv add "ai[otel]"`'
    ) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

_MESSAGES_ADAPTER = pydantic.TypeAdapter(list[messages_.Message])


def _messages_json(messages: list[messages_.Message]) -> str:
    # turn messages into a string to put it into an otel attribute
    return _MESSAGES_ADAPTER.dump_json(messages).decode()


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
                # otel only allows scalar attribute values or lists of them
                attrs[key] = (
                    value
                    if isinstance(value, str | bool | int | float)
                    else repr(value)
                )
    return attrs


def _derive_trace_id(id_: str) -> int:
    """Derive a stable otel trace id (128 bits) from one of ours."""
    return int.from_bytes(hashlib.sha256(id_.encode()).digest()[:16])


def _derive_span_id(id_: str) -> int:
    """Derive a stable otel span id (64 bits) from one of ours."""
    return int.from_bytes(hashlib.sha256(id_.encode()).digest()[:8])


# otel spec requires ids to be always minted by the otel sdk. we can't have
# that because of the durable execution case, where we can't always emit the
# span right where we created it. that's why we're using a smuggling hack to
# smuggle our ids in via a contextvar. fable is telling me that's a classic
# trick.
_smuggled_ids: contextvars.ContextVar[tuple[int, int] | None] = (
    contextvars.ContextVar("otel_smuggled_ids", default=None)
)


class _SmugglingIdGenerator(
    opentelemetry.sdk.trace.id_generator.RandomIdGenerator
):
    """Fake id generator for smuggling ids via a contextvar."""

    def __init__(
        self, inner: opentelemetry.sdk.trace.id_generator.IdGenerator
    ) -> None:
        self._inner = inner  # stock id generator

    def generate_trace_id(self) -> int:
        smuggled = _smuggled_ids.get()
        return (
            smuggled[0]
            if smuggled is not None
            else self._inner.generate_trace_id()
        )

    def generate_span_id(self) -> int:
        smuggled = _smuggled_ids.get()
        return (
            smuggled[1]
            if smuggled is not None
            else self._inner.generate_span_id()
        )


@runtime_checkable
class _Flushable(Protocol):
    """Duck type whether tracer provider can flush."""

    def force_flush(self) -> bool: ...
    def shutdown(self) -> None: ...


class OtelAdapter(telemetry.Adapter):
    """Maps framework spans onto otel spans."""

    def __init__(
        self,
        *,
        tracer_provider: opentelemetry.trace.TracerProvider | None = None,
    ) -> None:
        provider = tracer_provider or opentelemetry.trace.get_tracer_provider()
        self._provider = provider
        self._live: dict[str, opentelemetry.trace.Span] = {}

        self._is_smuggling_ids = False
        # printed a warning once when failing to smuggle ids
        self._warning_issued = False

        # swap TracerProvider's id generator with a smuggling one
        if isinstance(provider, opentelemetry.sdk.trace.TracerProvider):
            provider.id_generator = _SmugglingIdGenerator(provider.id_generator)
            self._is_smuggling_ids = True

        self._tracer = opentelemetry.trace.get_tracer(
            "ai", tracer_provider=provider
        )

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
        """Flush the provider's exporters."""
        # otel *api protocol* doesn't assume the tracer provider would have
        # a buffer, so it doesn't declare a flush method.
        # otel *sdk's* tracer provider DOES have a buffer, and therefore needs
        # to flush. users may also pass their own tracer providers modelled
        # after the stock one.
        if isinstance(self._provider, _Flushable):
            self._provider.force_flush()

    def shutdown(self) -> None:
        """Flush and stop the provider; spans pushed after this are lost."""
        if isinstance(self._provider, _Flushable):
            self._provider.shutdown()

    def _parent_context(
        self, span_: telemetry.Span
    ) -> opentelemetry.context.Context | None:
        # sometimes we need to set a parent span that isn't currently live.
        if span_.parent_id is None or span_.parent_id in self._live:
            # here the parent span is actually live, proceed as normal
            return None

        # parent is indeed not live
        # opentelemetry.io/docs/languages/python/cookbook/
        # 1. wrap the ids in a ``SpanContext``
        # 2. wrap that in a ``NonRecordingSpan``
        # 3. set it in a *fresh* ``Context()``

        if not self._is_smuggling_ids and not self._warning_issued:
            # trying to restore a parent when not smuggling ids
            # will result in a broken tree
            self._warning_issued = True
            logger.warning(
                "otel adapter: continuing a trace from another process "
                "needs an SDK tracer provider to control span ids; ids "
                "will not line up across processes"
            )
        return opentelemetry.trace.set_span_in_context(
            opentelemetry.trace.NonRecordingSpan(
                opentelemetry.trace.SpanContext(
                    trace_id=_derive_trace_id(span_.trace_id),
                    span_id=_derive_span_id(span_.parent_id),
                    is_remote=True,  # marks the parent as living elsewhere
                    trace_flags=opentelemetry.trace.TraceFlags(
                        opentelemetry.trace.TraceFlags.SAMPLED
                    ),
                )
            ),
            opentelemetry.context.Context(),
        )

    async def wrap_span(
        self, span_: telemetry.Span, /
    ) -> AsyncGenerator[None, Any]:
        parent_context = self._parent_context(span_)

        # convert framework's span ids into otel's and prepare for smuggling
        smuggled_token = _smuggled_ids.set(
            (_derive_trace_id(span_.trace_id), _derive_span_id(span_.id))
        )
        try:
            otel_span = self._tracer.start_span(
                self.span_name(span_),
                context=parent_context,
                start_time=span_.started_at,
            )
        finally:
            _smuggled_ids.reset(smuggled_token)

        self._live[span_.id] = otel_span

        try:
            # set live span as otel's current, so raw spans user opens in their
            # code can nest under it.
            with (
                opentelemetry.trace.use_span(
                    otel_span,
                    end_on_exit=False,  # we end it ourselves below
                    # errors come it via span_.error, not via raising, because
                    # we need them to serialize
                    record_exception=False,
                    set_status_on_exception=False,
                )
                if span_.set_as_current
                else contextlib.nullcontext()
            ):
                # span end resumes with None
                while (ev := (yield)) is not None:
                    otel_span.add_event(
                        ev.name,
                        {
                            k: v  # squash everything into scalars
                            if isinstance(v, str | bool | int | float)
                            else repr(v)
                            for k, v in ev.attributes.items()
                        },
                        timestamp=ev.time_ns,
                    )
        finally:
            self._live.pop(span_.id, None)
            for key, value in self.span_attributes(span_).items():
                otel_span.set_attribute(key, value)
            if span_.error is not None:
                otel_span.set_status(
                    opentelemetry.trace.StatusCode.ERROR,
                    f"{span_.error.type}: {span_.error.message}",
                )
            otel_span.end(end_time=span_.ended_at)


def install(
    *, tracer_provider: opentelemetry.trace.TracerProvider | None = None
) -> OtelAdapter:
    """Create the otel adapter, register it, and return it.

    Uses the global tracer provider unless one is passed.
    """
    adapter = OtelAdapter(tracer_provider=tracer_provider)
    telemetry.register(adapter)
    return adapter
