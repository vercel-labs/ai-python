"""OpenTelemetry adapter.

Experimental: not part of the stable API, may change or be removed.

::

    from ai.experimental_telemetry import otel
    otel.install()  # uses the global TracerProvider

Follows the ``gen_ai`` semantic conventions.

There's a certain amount of jank in this adapter, because we want it to work in
the durable execution case, which conflicts with the OTel spec.

Message content (``gen_ai.input.messages``, ``gen_ai.output.messages``,
``gen_ai.system_instructions``, ``gen_ai.tool.definitions``, tool call
arguments and results) is Opt-In in the conventions and off by default:
pass ``capture_content=True`` or set the standard
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment
variable (``true``, ``span_only``, or ``span_and_event``) to emit it.
Content is emitted in the semconv message shape (role + typed ``parts``
as a JSON string), not the framework's native message model.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .. import errors
from .. import experimental_telemetry as telemetry

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

    from ..types import messages as messages_
    from ..types import usage as usage_

logger = logging.getLogger(__name__)


CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def _capture_content_from_env() -> bool:
    value = os.environ.get(CAPTURE_CONTENT_ENV, "")
    return value.strip().lower() in ("true", "span_only", "span_and_event")


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


def _kind(sp: telemetry.Span) -> opentelemetry.trace.SpanKind:
    # Inference spans SHOULD be CLIENT per semconv; agent and tool
    # spans run in-process and stay INTERNAL.
    match sp.data:
        case telemetry.AiStreamSpanData() | telemetry.AiGenerateSpanData():
            return opentelemetry.trace.SpanKind.CLIENT
        case _:
            return opentelemetry.trace.SpanKind.INTERNAL


def _json_or_raw(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _semconv_part(part: dict[str, Any]) -> dict[str, Any]:
    """Map one dumped message part onto its gen_ai semconv shape.

    Shapes per ``docs/gen-ai/non-normative/models.py`` in the semconv
    repo.  Unknown kinds degrade to the catch-all generic part, which
    carries only ``type``.
    """
    match part["kind"]:
        case "text":
            return {"type": "text", "content": part["text"]}
        case "reasoning":
            return {"type": "reasoning", "content": part["text"]}
        case "tool_call":
            return {
                "type": "tool_call",
                "id": part["tool_call_id"],
                "name": part["tool_name"],
                "arguments": _json_or_raw(part["tool_args"]),
            }
        case "tool_result":
            return {
                "type": "tool_call_response",
                "id": part["tool_call_id"],
                "response": part["result"],
            }
        case "builtin_tool_call":
            return {
                "type": "server_tool_call",
                "id": part["tool_call_id"],
                "name": part["tool_name"],
                "server_tool_call": {
                    "type": part["tool_name"],
                    "arguments": _json_or_raw(part["tool_args"]),
                },
            }
        case "builtin_tool_return":
            return {
                "type": "server_tool_call_response",
                "id": part["tool_call_id"],
                "server_tool_call_response": {
                    "type": part["tool_name"],
                    "response": part["result"],
                },
            }
        case "file":
            media_type = part["media_type"]
            modality = media_type.split("/")[0]
            if modality not in ("image", "video", "audio"):
                modality = "document"
            # data is a str after the JSON dump: a URL or base-64
            # content.
            if part["data"].startswith(("http://", "https://")):
                return {
                    "type": "uri",
                    "modality": modality,
                    "mime_type": media_type,
                    "uri": part["data"],
                }
            return {
                "type": "blob",
                "modality": modality,
                "mime_type": media_type,
                "content": part["data"],
            }
        case kind:
            return {"type": kind}


def _semconv_parts(
    message: messages_.Message,
) -> tuple[str, list[dict[str, Any]]]:
    dumped = message.model_dump(mode="json", fallback=str)
    return dumped["role"], [_semconv_part(p) for p in dumped["parts"]]


def _content_attributes(
    messages: list[messages_.Message],
    output: messages_.Message | None,
    *,
    error: bool,
    finish_reason: str | None = None,
) -> dict[str, str]:
    """gen_ai message content attributes, in the semconv message shape.

    System messages are excluded from ``input.messages`` and carried as
    ``gen_ai.system_instructions`` (a flat parts list), per semconv.
    The schema requires ``finish_reason`` on output messages; when the
    span data doesn't carry one (replay, spans recorded before capture)
    it is inferred: ``error`` when the span errored, ``tool_call`` when
    the output requests a tool call, ``stop`` otherwise.
    """
    system_parts: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for message in messages:
        role, parts = _semconv_parts(message)
        if role == "system":
            system_parts += parts
        else:
            inputs.append({"role": role, "parts": parts})
    attrs = {"gen_ai.input.messages": json.dumps(inputs)}
    if system_parts:
        attrs["gen_ai.system_instructions"] = json.dumps(system_parts)
    if output is not None:
        role, parts = _semconv_parts(output)
        if finish_reason is not None:
            finish = finish_reason
        elif error:
            finish = "error"
        elif any(p["type"] == "tool_call" for p in parts):
            finish = "tool_call"
        else:
            finish = "stop"
        attrs["gen_ai.output.messages"] = json.dumps(
            [{"role": role, "parts": parts, "finish_reason": finish}]
        )
    return attrs


def _field(obj: Any, name: str) -> Any:
    # Params are ``Any`` on span data: live params objects in-process,
    # plain dicts after a JSON round-trip.  Read both.
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _has_field(obj: Any, name: str) -> bool:
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


_SEMCONV_SAMPLER_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)


def _request_attributes(params: Any) -> dict[str, Any]:
    """``gen_ai.request.*`` attributes from inference params.

    Only numeric, explicitly-set values come through: provider-default
    sentinels and framework knobs with no semconv name (``min_p``,
    ``repetition_penalty``, ...) are skipped.
    """
    attrs: dict[str, Any] = {}
    sampling = _field(params, "sampling")
    if isinstance(sampling, dict):
        for sampler in sampling.values():
            for name in _SEMCONV_SAMPLER_FIELDS:
                value = _field(sampler, name)
                if isinstance(value, int | float) and not isinstance(
                    value, bool
                ):
                    attrs[f"gen_ai.request.{name}"] = value
    max_tokens = _field(_field(params, "output"), "max_tokens")
    if isinstance(max_tokens, int):
        attrs["gen_ai.request.max_tokens"] = max_tokens
    effort = _field(_field(params, "reasoning"), "effort")
    if isinstance(effort, str):
        attrs["gen_ai.request.reasoning.level"] = effort
    return attrs


def _usage_attributes(usage: usage_.Usage) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.usage.input_tokens": usage.input_tokens,
        "gen_ai.usage.output_tokens": usage.output_tokens,
    }
    if usage.reasoning_tokens is not None:
        attrs["gen_ai.usage.reasoning.output_tokens"] = usage.reasoning_tokens
    if usage.cache_read_tokens is not None:
        attrs["gen_ai.usage.cache_read.input_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens is not None:
        attrs["gen_ai.usage.cache_creation.input_tokens"] = (
            usage.cache_write_tokens
        )
    return attrs


def _provider_name(model: str, provider: str | None) -> str | None:
    # Gateway model ids are "provider/model"; the prefix names the
    # actual inference provider and matches semconv's well-known values
    # better than the framework provider adapter's name.
    if "/" in model:
        return model.partition("/")[0]
    return provider


def _time_to_first_chunk(sp: telemetry.Span) -> float | None:
    if sp.started_at is None:
        return None
    for event in sp.events:
        if event.name == telemetry.FIRST_TOKEN:
            return (event.time_ns - sp.started_at) / 1e9
    return None


def _tool_definitions(names: list[str]) -> str:
    return json.dumps([{"type": "function", "name": n} for n in names])


def _attributes(sp: telemetry.Span, *, capture_content: bool) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if sp.replay:
        attrs["ai.replay"] = True
    match sp.data:
        case telemetry.AiStreamSpanData() as d:
            attrs["gen_ai.operation.name"] = "chat"
            provider = _provider_name(d.model, d.provider)
            if provider is not None:
                attrs["gen_ai.provider.name"] = provider
            attrs["gen_ai.request.model"] = d.model
            attrs["gen_ai.request.stream"] = True
            if d.output_type is not None:
                # Structured output was requested; the semconv value is
                # the output kind, not the Python type name.
                attrs["gen_ai.output.type"] = "json"
            attrs |= _request_attributes(d.params)
            ttfc = _time_to_first_chunk(sp)
            if ttfc is not None:
                attrs["gen_ai.response.time_to_first_chunk"] = ttfc
            if d.finish_reason is not None:
                attrs["gen_ai.response.finish_reasons"] = [d.finish_reason]
            if d.response_id is not None:
                attrs["gen_ai.response.id"] = d.response_id
            if d.response_model is not None:
                attrs["gen_ai.response.model"] = d.response_model
            if d.usage is not None:
                attrs |= _usage_attributes(d.usage)
            if capture_content:
                if d.tool_names:
                    attrs["gen_ai.tool.definitions"] = _tool_definitions(
                        d.tool_names
                    )
                attrs |= _content_attributes(
                    d.messages,
                    d.message,
                    error=sp.error is not None,
                    finish_reason=d.finish_reason,
                )
        case telemetry.AiGenerateSpanData() as d:
            attrs["gen_ai.operation.name"] = "generate_content"
            provider = _provider_name(d.model, d.provider)
            if provider is not None:
                attrs["gen_ai.provider.name"] = provider
            attrs["gen_ai.request.model"] = d.model
            n = _field(d.params, "n")
            if isinstance(n, int) and n != 1:
                attrs["gen_ai.request.choice.count"] = n
            seed = _field(d.params, "seed")
            if isinstance(seed, int):
                attrs["gen_ai.request.seed"] = seed
            if d.params is not None:
                # ImageParams vs VideoParams: only the latter has fps.
                attrs["gen_ai.output.type"] = (
                    "video" if _has_field(d.params, "fps") else "image"
                )
            if d.usage is not None:
                attrs |= _usage_attributes(d.usage)
            if capture_content:
                attrs |= _content_attributes(
                    d.messages, d.message, error=sp.error is not None
                )
        case telemetry.ToolExecutionSpanData() as d:
            attrs["gen_ai.operation.name"] = "execute_tool"
            attrs["gen_ai.tool.name"] = d.tool_name
            attrs["gen_ai.tool.type"] = "function"
            attrs["gen_ai.tool.call.id"] = d.tool_call_id
            if d.tool_description is not None:
                attrs["gen_ai.tool.description"] = d.tool_description
            if d.is_error:
                # The exception type is not captured when the error is
                # only a model-facing result (see NEW_DATA_CAPTURE.md);
                # ``_OTHER`` is semconv's fallback class.  Overwritten
                # with the real type when the span carries an error.
                attrs["error.type"] = "_OTHER"
            if capture_content:
                if d.args is not None:
                    attrs["gen_ai.tool.call.arguments"] = json.dumps(
                        d.args, default=str
                    )
                if d.result is not None:
                    attrs["gen_ai.tool.call.result"] = json.dumps(
                        d.result, default=str
                    )
        case telemetry.RunSpanData() as d:
            attrs["gen_ai.operation.name"] = "invoke_agent"
            attrs["gen_ai.agent.name"] = d.agent
            provider = _provider_name(d.model, d.provider)
            if provider is not None:
                attrs["gen_ai.provider.name"] = provider
            attrs["gen_ai.request.model"] = d.model
            if d.output_type is not None:
                # Structured output was requested; the semconv value is
                # the output kind, not the Python type name.
                attrs["gen_ai.output.type"] = "json"
            attrs |= _request_attributes(d.params)
            if d.usage is not None:
                attrs |= _usage_attributes(d.usage)
            if capture_content:
                if d.tool_names:
                    attrs["gen_ai.tool.definitions"] = _tool_definitions(
                        d.tool_names
                    )
                attrs |= _content_attributes(
                    d.messages, d.final_message, error=sp.error is not None
                )
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
        capture_content: bool | None = None,
    ) -> None:
        provider = tracer_provider or opentelemetry.trace.get_tracer_provider()
        self._provider = provider
        self._live: dict[str, opentelemetry.trace.Span] = {}

        self._capture_content = (
            _capture_content_from_env()
            if capture_content is None
            else capture_content
        )

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
        return _attributes(span_, capture_content=self._capture_content)

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
                kind=_kind(span_),
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
                otel_span.set_attribute("error.type", span_.error.type)
                otel_span.set_status(
                    opentelemetry.trace.StatusCode.ERROR,
                    f"{span_.error.type}: {span_.error.message}",
                )
            otel_span.end(end_time=span_.ended_at)


def install(
    *,
    tracer_provider: opentelemetry.trace.TracerProvider | None = None,
    capture_content: bool | None = None,
) -> OtelAdapter:
    """Create the otel adapter, register it, and return it.

    Uses the global tracer provider unless one is passed.  Message
    content capture is off by default, per the gen_ai conventions;
    ``capture_content=True`` (or the
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment
    variable) turns it on.
    """
    adapter = OtelAdapter(
        tracer_provider=tracer_provider, capture_content=capture_content
    )
    telemetry.register(adapter)
    return adapter
