"""Telemetry API: spans, sinks, and adapters.

Experimental: not part of the stable API, may change or be removed.

:class:`Span` is a record of work. It carries ids, timestamps, parent id, and
work-specific data::

    async with ai.experimental_telemetry.span("retrieval") as sp:
        sp.set_attrs(query=q)
        docs = await search(q)
        sp.set_attrs(count=len(docs))

Nesting is automatic: the current span is tracked using a context var.

In cases where the high-level API doesn't work (e.g. durable execution),
selectively using low-level API may be a better fit.

1. make a span using `create_span`
2. set `.started_at`, `ended_at`, and necessary data fields
3. push updates to the sink one or more times

:meth:`Span.push` delivers a snapshot of the span to the current *sink*.
The default sink is the adapter registry, meaning on push all registered
adapters will get the updated version of the span::

    sp = create_span("turn")                 # identity only, reports nothing
    sp.set_attrs(session=sid)
    sp.stamp_start()                         # writes started_at
    await sp.push()                          # visible to sinks/adapters
    ...                                      # possibly elsewhere, later:
    await sp.stamp_end().push()              # complete

Because ``Span`` is a pydantic model it crosses process boundaries like
any other data (``model_dump`` / ``model_validate``); you can restore it
and set it as current using :func:`use_span` or pass to a child span explicity.

An adapter processes spans and decides what to do with them::

    class MyAdapter:
        async def on_span_start(self, span): ...
        async def on_span_end(self, span): ...
        async def on_span_event(self, span, event): ...

    ai.experimental_telemetry.register(MyAdapter())

Adapters dispatch on the type of ``span.data``.  An adapter that crashes
is logged and skipped, it never kills the run.

:func:`adapter` builds an adapter from an async generator function or
class, using the pytest fixture-style trick.

A :class:`Sink` is what sits *before* the adapters and routes span snapshots.
It exists, again, because of the durable execution case: the user can control
whether spans in the current context get sent to telemetry adapters, collected
into a list to be pushed later, or discarded.

:func:`use_sink` sets the current sink.

If telemetry is off, i.e. no adapters registered and no sink used, the module
doesn't use random or read the clock.

A :class:`SpanEvent` is a named, timestamped milestone inside a span's
lifetime (``first_token``, ``hook_resolved``, ...): append it to
``span.events`` and push.

Span timestamps come from the ambient clock (:func:`now_ns`);
:func:`use_time` overrides it per-context, e.g. with a deterministic
clock inside a durable workflow.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import logging
import time
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    Self,
    cast,
    overload,
    runtime_checkable,
)

import pydantic

# ``typing.TypeVar`` lacks the ``default=`` kwarg on Python <3.13.
# Use the typing_extensions backport so this works on 3.12 too.
from typing_extensions import TypeVar

from ..types import messages as messages_
from ..types import usage as usage_

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Awaitable,
        Callable,
        Iterable,
        Mapping,
    )

    from ..models.core import params as params_

    # 1. RunSpanData and friends are pydantic models. RunSpanData.params is
    #    typed as InferenceRequestParams.
    # 2. InferenceRequestParams.sampling (and other fields) have
    #    a ModelProviderDefault annotation.
    # 3. ModelProviderDefault is a plain setinel class that causes pydantic to
    #    freak out when building a schema.

    # also, importing without a guard would cause a circular import
    # in models/core/api.py
    _InferenceParams = params_.InferenceRequestParams
    _GenerateParams = params_.GenerateParams
else:
    _InferenceParams = Any
    _GenerateParams = Any

logger = logging.getLogger(__name__)


# this api is for plugging a deterministic clock in, so that the sdk
# could be used inside a deterministic workflow body

_span_clock: contextvars.ContextVar[Callable[[], int] | None] = (
    contextvars.ContextVar("span_clock", default=None)
)


def now_ns() -> int:
    """Nanoseconds since the epoch, from the ambient span clock.

    The wall clock by default; whatever :func:`use_time` installed
    otherwise.
    """
    clock = _span_clock.get()
    return clock() if clock is not None else time.time_ns()


@contextlib.asynccontextmanager
async def use_time(now_ns: Callable[[], int]) -> AsyncIterator[None]:
    """Read span timestamps from ``now_ns`` within this context.

    Framework's observability creates timestamps. This API can be
    used to plug an approved clock function in durable execution
    settings::

        async with ai.experimental_telemetry.use_time(workflow.time_ns):
            ...  # spans opened here read time from workflow.time_ns

    This can also be used as a decorator on async functions::

        @ai.experimental_telemetry.use_time(clock.time_ns)
        async def run(...):
            ...
    """
    token = _span_clock.set(now_ns)
    try:
        yield
    finally:
        _span_clock.reset(token)


# span data types
# the data type tells you what kind of span it is.


class SpanData(Protocol):
    """Anything with a ``kind`` is span data.

    Implement it with a pydantic model to make your own typed spans.

        class RetrievalSpanData(pydantic.BaseModel):
            kind: Literal["retrieval"] = "retrieval"
            query: str
            count: int | None = None

        data = RetrievalSpanData(query=q)
        async with ai.experimental_telemetry.span(data) as sp:
            docs = await search(q)
            sp.data.count = len(docs)  # typed

    ``kind`` is what travels with the serialized data. Restore with
    ``Span[RetrievalSpanData].model_validate(...)`` to get it typed, otherwise
    you'll get a plain dict.
    """

    @property
    def kind(self) -> str: ...


class RunSpanData(pydantic.BaseModel):
    """One ``Agent.run``: the whole loop.

    ``blocked``/``final_message``/``usage`` are set at span end:
    ``blocked`` is True when the run ended suspended on an unresolved
    hook (see ``AgentStream.blocked``), ``final_message`` is the last
    assistant message produced, if any, ``usage`` is the sum of usage
    across the messages the run produced.
    """

    kind: Literal["run"] = "run"
    agent: str
    model: str
    messages: list[messages_.Message]
    provider: str | None = None
    tool_names: list[str] | None = None
    output_type: str | None = None
    params: _InferenceParams | None = None
    blocked: bool = False
    final_message: messages_.Message | None = None
    usage: usage_.Usage | None = None


class LoopTurnSpanData(pydantic.BaseModel):
    """One turn of the default agent loop.

    Carries no fields — it exists so adapters can group a turn's model
    and tool spans.  Turn order is given by ``started_at``/``parent_id``.
    """

    kind: Literal["loop_turn"] = "loop_turn"


class AiStreamSpanData(pydantic.BaseModel):
    """One streaming LLM call."""

    kind: Literal["ai_stream"] = "ai_stream"
    model: str
    messages: list[messages_.Message]
    params: _InferenceParams | None = None
    provider: str | None = None
    tool_names: list[str] | None = None
    output_type: str | None = None
    message: messages_.Message | None = None
    usage: usage_.Usage | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    response_model: str | None = None


class AiGenerateSpanData(pydantic.BaseModel):
    """One non-streaming generation call (images, video, ...).

    ``message``/``usage`` are set at span end.
    """

    kind: Literal["ai_generate"] = "ai_generate"
    model: str
    messages: list[messages_.Message]
    params: _GenerateParams | None = None
    provider: str | None = None
    message: messages_.Message | None = None
    usage: usage_.Usage | None = None


class ToolExecutionSpanData(pydantic.BaseModel):
    """One tool execution, from dispatch to result.

    ``model_input`` is the value the LLM sees on its next turn, set
    only when it differs from ``result`` (aggregator-backed tools).
    ``tool_description`` is the tool's model-facing description,
    stamped at dispatch.
    """

    kind: Literal["tool_execution"] = "tool_execution"
    tool_name: str
    tool_call_id: str
    tool_description: str | None = None
    args: dict[str, Any] | None = None
    result: Any = None
    model_input: Any = None
    is_error: bool = False


class HookSpanData(pydantic.BaseModel):
    """One hook suspension, from deferred until resolved or cancelled.

    ``tool_call_id`` links the hook to the tool call it suspends, if
    any (e.g. approval gating).  ``resolution`` carries the resolution
    data when the hook resolves.
    """

    kind: Literal["hook"] = "hook"
    label: str
    hook_type: str
    metadata: dict[str, Any]
    tool_call_id: str | None = None
    status: Literal["pending", "resolved", "cancelled"] = "pending"
    resolution: dict[str, Any] | None = None


class CustomSpanData(pydantic.BaseModel):
    """A user span made with ``span("name")``.

    Attributes are set with :meth:`Span.set_attrs`.
    """

    kind: Literal["custom"] = "custom"
    attrs: dict[str, Any]


# names for span events shared between producers,
# adapters, and tests
FIRST_TOKEN = "first_token"
RESPONSE_COMPLETE = "response_complete"
HOOK_DEFERRED = "hook_deferred"
HOOK_RESOLVED = "hook_resolved"
HOOK_CANCELLED = "hook_cancelled"


class SpanEvent(pydantic.BaseModel):
    """A named, timestamped milestone inside a span's lifetime.

    Lives in ``span.events``; the next push delivers it.
    :meth:`Span.add_event` is the shorthand for the common case::

        sp.add_event(FIRST_TOKEN)
        await sp.push()

    Ordering: events on one span are delivered in list order; there is
    no ordering guarantee across spans.
    """

    name: str
    time_ns: int
    attrs: dict[str, Any]


class SpanError(pydantic.BaseModel):
    """A serializable record of the failure that ended a span.

    Spans cross process boundaries, so the error they carry is plain
    data, not a live exception.
    """

    type: str
    message: str

    @classmethod
    def from_exception(cls, exc: BaseException) -> SpanError:
        return cls(type=type(exc).__name__, message=str(exc))


_FrameworkData = Annotated[
    RunSpanData
    | LoopTurnSpanData
    | AiStreamSpanData
    | AiGenerateSpanData
    | ToolExecutionSpanData
    | HookSpanData
    | CustomSpanData,
    pydantic.Field(discriminator="kind"),
]

# adapters only read the data, they don't swap it out, so it's safe to make it
# covariant.
if TYPE_CHECKING:
    DataT_co = TypeVar(
        "DataT_co", bound=SpanData, default=SpanData, covariant=True
    )
else:
    # try to deserialize to _FrameworkData and fallback to Any
    DataT_co = TypeVar("DataT_co", default=_FrameworkData | Any)
# covariant type can't be used for a function's input parameter, and we need
# to type span()
DataT = TypeVar("DataT", bound=SpanData)


class Span(pydantic.BaseModel, Generic[DataT_co]):
    """A serializable record of a unit of work.

    Generic in its data type: ``span(RetrievalSpanData(...))`` gives a
    ``Span[RetrievalSpanData]``, so late assignments to ``sp.data``
    fields are type checked.  A bare ``Span`` is ``Span[SpanData]``.

    Lifecycle is encoded in timestamp fields:

    - if ``started_at=None``, this span hasn't started,
    - if ``ended_at`` is set, the span is complete.

    Mutate fields, then :meth:`push` to report the new state. Nothing is
    reported except by pushing.

    A span round-trips through ``model_dump``/``model_validate`` like
    any pydantic model. Use ``Span[MyData].model_validate(...)`` to restore
    custom data types.

    ``replay=True`` marks work that is being replayed (resume,
    serverless re-entry) rather than performed live.

    A span created while telemetry was off (see :func:`is_enabled`) has an
    empty ``id`` and is a noop: :meth:`push` delivers nothing.

    ``set_as_current=False`` marks a span that does not set itself
    as a current. This is used by ai.stream's span that stays open for
    the duration of a loop turn because of the context manager api.
    Adapters must not make those spans "current" either to avoid
    nesting discrepancies.

    ``trace_attrs`` is user data that propagates down from parent to child
    on creation.

    ``schema_version`` tracks the shape of spans and their data types.
    """

    name: str
    data: DataT_co
    id: str
    trace_id: str
    parent_id: str | None = None
    started_at: int | None = None  # ns since the epoch, see ``now_ns``
    ended_at: int | None = None
    error: SpanError | None = None
    replay: bool = False
    set_as_current: bool = True
    events: list[SpanEvent] = pydantic.Field(default_factory=list)
    trace_attrs: dict[str, Any] = pydantic.Field(default_factory=dict)

    schema_version: ClassVar[int] = 5

    def set_attrs(
        self, attrs: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> None:
        """Attach attributes to a span created with ``span("name")``.

        Attribute names that aren't valid Python keywords (viewers use
        dotted names like ``"output.value"``) go in the positional
        mapping; it merges with the keyword arguments::

            sp.set_attrs({"output.value": title}, model="haiku")
        """
        if not isinstance(self.data, CustomSpanData):
            raise TypeError(
                "set_attrs() only works on user spans; framework "
                "spans carry typed data, assign its fields directly"
            )
        self.data.attrs.update({**(attrs or {}), **kwargs})

    def add_event(
        self,
        name: str,
        attrs: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> SpanEvent:
        """Append a named event stamped with the ambient clock.

        Appends only; the event is delivered by the next :meth:`push`, or when
        exiting the ``span()`` context manager::

            sp.add_event(FIRST_TOKEN)
            await sp.push()

        Returns the appended event.
        """
        event = SpanEvent(
            name=name,
            # a noop span (telemetry off at creation) reads no clock
            time_ns=now_ns() if self.id else 0,
            attrs={**(attrs or {}), **kwargs},
        )
        self.events.append(event)
        return event

    def stamp_start(self) -> Self:
        """Set ``started_at`` from the ambient clock; return the span.

        Only writes the field; nothing is reported until :meth:`push`.
        Returns the span so minting reads as one expression::

            payload = create_span("turn").stamp_start().model_dump()
        """
        self.started_at = now_ns()
        return self

    def stamp_end(
        self, *, error: SpanError | BaseException | None = None
    ) -> Self:
        """Set ``ended_at`` from the ambient clock; return the span.

        Also records ``error`` when one is given by converting an exception
        to a serializable :class:`SpanError`.  ``error=None``
        keeps any error already on the span.  Only writes fields;
        push to report::

            await sp.stamp_end(error=exc).push()
        """
        self.ended_at = now_ns()
        if isinstance(error, BaseException):
            self.error = SpanError.from_exception(error)
        elif error is not None:
            self.error = error
        return self

    async def push(self) -> None:
        """Deliver a frozen copy of this span to the current sink.

        The default sink drives the registered adapters (see
        :func:`register`); :func:`use_sink` reroutes pushes within a
        context.  Each push snapshots the whole span, so keep mutating
        and push again as the work progresses. The last push with
        ``ended_at`` set is the complete record.

        Telemetry never kills the run: a sink (or adapter) failure is
        logged and swallowed.
        """
        if not self.id:
            return  # noop: telemetry was off at creation
        sink = _current_sink.get() or _registry_sink
        try:
            await sink.on_push(self.model_copy(deep=True))
        except Exception:
            logger.exception("telemetry sink %r raised in on_push", sink)


# utilities for managing the current span

_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)


def current_span() -> Span | None:
    """Return the current span, or ``None`` when no span is open."""
    return _current.get()


@contextlib.asynccontextmanager
async def use_span(span_: Span | None) -> AsyncIterator[None]:
    """Make ``span_`` the current span within this context.

    Unlike :func:`span` this is pure context plumbing with no timestamps
    and no pushes.  Use it to continue a trace around existing work, e.g.
    parenting under a span restored from another process::

        turn_span = Span.model_validate(payload["turn_span"])
        async with ai.experimental_telemetry.use_span(turn_span):
            ...  # spans opened here parent under turn_span

    ``None`` is a no-op.
    """
    if span_ is None:
        yield
        return
    token = _current.set(span_)
    try:
        yield
    finally:
        _current.reset(token)


# sinks are used to control where spans go within the current context.
# that could be the adapter registry (so they immediately get sent to the
# observability backend; or it could be a DictSink that allows us to defer
# sending them until later.


class Sink(Protocol):
    """Anything that accepts pushed span snapshots."""

    async def on_push(self, span: Span, /) -> None: ...


_current_sink: contextvars.ContextVar[Sink | None] = contextvars.ContextVar(
    "current_sink", default=None
)


@contextlib.asynccontextmanager
async def use_sink(sink: Sink | None) -> AsyncIterator[None]:
    """Route span pushes to ``sink`` within this context.

    The default (outside any ``use_sink``) is the adapter registry.
    Inside a durable workflow body where side effects are not allowed, you
    can route to a :class:`DictSink` instead and re-push the collected spans
    from a step / activity.

    ``None`` is a no-op.
    """
    if sink is None:
        yield
        return
    token = _current_sink.set(sink)
    try:
        yield
    finally:
        _current_sink.reset(token)


class DictSink:
    """A sink that keeps the latest snapshot of every span pushed to it.

    ``spans`` maps span id to the most recent snapshot, in first-push
    order.  Scoop the complete ones out as data (:attr:`finished_spans`) and
    re-push them where the real sink is available::

        sink = DictSink()
        async with use_sink(sink):
            ...  # replayed / suspendable code
        payload = [s.model_dump(mode="json") for s in sink.finished_spans]

        # elsewhere (a workflow step, another process):
        await push_all(payload)
    """

    def __init__(self) -> None:
        self.spans: dict[str, Span] = {}

    async def on_push(self, span: Span, /) -> None:
        self.spans[span.id] = span

    @property
    def finished_spans(self) -> list[Span]:
        """The collected spans that have ended, in first-push order."""
        return [s for s in self.spans.values() if s.ended_at is not None]


async def push_all(spans: Iterable[Span | Mapping[str, Any]]) -> None:
    """Push each span, in order; dumped spans are validated first.

    This is a utility that can be used with :class:`DictSink` to push
    a bunch of collected spans all at once::

        await push_all(payload)
    """
    for item in spans:
        span = item if isinstance(item, Span) else Span.model_validate(item)
        await span.push()


class _RegistrySink:
    """The default sink: drives registered adapters.

    Translates a stream of span snapshots into
    ``on_span_start`` / ``on_span_event`` / ``on_span_end`` callbacks.

    Rules:

    - a snapshot with ``started_at=None`` reports nothing (the span
      hasn't started);
    - the first snapshot of an id fires ``on_span_start``, then
      ``on_span_event`` per event, then ``on_span_end`` if the snapshot
      is already complete; so a span that lived elsewhere and arrives
      finished is delivered whole;
    - later snapshots fire ``on_span_event`` for events not seen
      before, and ``on_span_end`` once ``ended_at`` appears;
    - after the end the id is forgotten: pushing a completed span again
      re-delivers it in full.
    """

    def __init__(self) -> None:
        self._views: dict[str, Span] = {}

    async def on_push(self, span: Span, /) -> None:
        if span.started_at is None:
            return
        view = self._views.get(span.id)
        if view is None:
            view = span
            self._views[view.id] = view
            await _dispatch(lambda a: a.on_span_start(view))
            fresh = list(view.events)
        else:
            seen = len(view.events)
            # we need to preserve the view's identity
            # but update all of the fields
            view.__dict__.update(span.__dict__)
            fresh = view.events[seen:]
        for event in fresh:
            # the default binds `event` now; a plain lambda would trip B023
            def call(
                a: AdapterProtocol, ev: SpanEvent = event
            ) -> Awaitable[None]:
                return a.on_span_event(view, ev)

            await _dispatch(call)
        if view.ended_at is not None:
            del self._views[view.id]
            await _dispatch(lambda a: a.on_span_end(view))


_registry_sink = _RegistrySink()


# adapter registry utilities


@runtime_checkable
class AdapterProtocol(Protocol):
    """The async callbacks implemented by a telemetry adapter."""

    async def on_span_start(self, span: Span, /) -> None: ...

    async def on_span_event(self, span: Span, event: SpanEvent, /) -> None: ...

    async def on_span_end(self, span: Span, /) -> None: ...


class AdapterCallable(Protocol):
    """The ``__call__`` shape the :func:`adapter` decorator accepts.

    Static-typing companion to :class:`AdapterProtocol`: checkers don't
    see the callbacks :func:`adapter` mixes into a decorated class, so
    :func:`register` accepts this shape as well.
    """

    def __call__(self, span: Span, /) -> AsyncGenerator[None, Any] | None: ...


_adapters: list[AdapterProtocol] = []


def register(adapter: AdapterProtocol | AdapterCallable) -> None:
    """Add an adapter.  Multiple adapters coexist independently."""
    if not isinstance(adapter, AdapterProtocol):
        raise TypeError(
            f"{adapter!r} does not implement the adapter callbacks; "
            "an adapter class must be decorated with @adapter"
        )
    _adapters.append(adapter)


def unregister(adapter: AdapterProtocol | AdapterCallable) -> None:
    """Remove a previously registered adapter."""
    assert isinstance(adapter, AdapterProtocol)  # register() enforced it
    _adapters.remove(adapter)


def is_enabled() -> bool:
    """Whether anything is listening for spans.

    True when a sink is routed with :func:`use_sink` or at least one
    adapter is registered.
    """
    return _current_sink.get() is not None or bool(_adapters)


async def _dispatch(
    call: Callable[[AdapterProtocol], Awaitable[None]],
) -> None:
    for adapter in list(_adapters):
        try:
            await call(adapter)
        except Exception:
            logger.exception("telemetry adapter %r raised", adapter)


# span building utilities


@overload
def create_span(
    name_or_data: str,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
) -> Span[CustomSpanData]: ...


@overload
def create_span(
    name_or_data: DataT,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
) -> Span[DataT]: ...


def create_span(
    name_or_data: str | SpanData,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
) -> Span[Any]:
    """Create a span: identity only, nothing is reported.

    Mints the span id, and takes trace id, parentage, and a copy of
    ``trace_attrs`` from ``parent`` (default: the current span; a fresh
    trace when there is none).  ``parent`` may be a span restored from
    JSON.

    The span has no timestamps yet: :meth:`Span.stamp_start` and
    :meth:`Span.push` when the work begins, or hand the whole
    lifecycle to the :func:`span` context manager.

    While telemetry is off (see :func:`is_enabled`) this returns a noop
    span instead.
    """
    if isinstance(name_or_data, str):
        name = name_or_data
        data: SpanData = CustomSpanData(attrs={})
    else:
        name = name_or_data.kind
        data = name_or_data
    if not is_enabled():
        return Span(
            name=name,
            data=data,
            id="",
            trace_id="",
            parent_id=None,
            replay=replay,
            set_as_current=False,
        )
    if parent is None:
        parent = _current.get()
    if parent is None:
        trace_id, parent_id = messages_.generate_id("trace"), None
        trace_attrs: dict[str, Any] = {}
    else:
        trace_id, parent_id = parent.trace_id, parent.id
        trace_attrs = dict(parent.trace_attrs)
    return Span(
        name=name,
        data=data,
        id=messages_.generate_id("span"),
        trace_id=trace_id,
        parent_id=parent_id,
        replay=replay,
        set_as_current=set_as_current,
        trace_attrs=trace_attrs,
    )


@overload
def span(
    name_or_data: str,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
) -> contextlib.AbstractAsyncContextManager[Span[CustomSpanData]]: ...


@overload
def span(
    name_or_data: DataT,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
) -> contextlib.AbstractAsyncContextManager[Span[DataT]]: ...


def span(
    name_or_data: str | SpanData,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
) -> contextlib.AbstractAsyncContextManager[Span[Any]]:
    """Open a span; it sets itself as current inside the block.

    Sugar over the data api: creates the span, stamps ``started_at``
    and pushes on enter; stamps ``ended_at`` (and ``error``, if the
    block raised) and pushes on exit.

    Pass a name for a user span (set attributes on it with
    :meth:`Span.set_attrs`), or a :class:`SpanData` instance
    for a typed one. Exceptions are recorded on the span and
    re-raised.

    ``parent`` overrides the ambient parent for this span: a live
    :class:`Span`, or one restored from another process to continue its
    trace here.  The default parents under the current span.

    ``set_as_current=False`` keeps the span from becoming current:
    work done while it is open parents to *its* parent instead. Used by
    ai.stream because of the context manager api.
    """
    # type checkers can't apply asynccontextmanager
    # to an overloaded function directly.
    return _span_impl(
        name_or_data,
        parent=parent,
        replay=replay,
        set_as_current=set_as_current,
    )


@contextlib.asynccontextmanager
async def _span_impl(
    name_or_data: str | SpanData,
    /,
    *,
    parent: Span | None,
    replay: bool,
    set_as_current: bool,
) -> AsyncIterator[Span[Any]]:
    sp: Span[Any] = create_span(
        name_or_data,
        parent=parent,
        replay=replay,
        set_as_current=set_as_current,
    )
    if not sp.id:
        # noop: no timestamps, no pushes, never current
        yield sp
        return
    sp.stamp_start()
    await sp.push()
    token = _current.set(sp) if set_as_current else None
    try:
        yield sp
    except BaseException as exc:
        # GeneratorExit is how a consumer closes a stream early.
        if not isinstance(exc, GeneratorExit) and sp.error is None:
            sp.error = SpanError.from_exception(exc)
        raise
    finally:
        # a span that set itself as current must be closed while it is
        # still current, in the task that opened it.
        misordered = False
        if token is not None:
            if _current.get() is not sp:
                misordered = True
            else:
                try:
                    _current.reset(token)
                except ValueError:
                    # token from a different task's context.
                    misordered = True
        sp.ended_at = now_ns()
        await sp.push()
        if misordered:
            raise RuntimeError(
                f"span {sp.name!r} closed out of order: it is not the "
                "current span at close, or was closed in a different "
                "task than it was opened in. A span that sets itself "
                "as current must be closed innermost-first in the "
                "opening task; open overlapping spans with "
                "set_as_current=False."
            )


class AdapterMixin(AdapterProtocol):
    """The driver :func:`adapter` mixes into a decorated class.

    Implements the adapter callbacks. Don't subclass it directly. Put
    :func:`adapter` over a class with ``__call__`` async generator method.
    """

    # Name-mangled so the driver's bookkeeping can never collide with
    # an attribute of the class it is mixed into; created lazily so
    # that class keeps its ``__init__`` entirely to itself.
    __live: dict[str, AsyncGenerator[None, Any]] | None = None

    # Positional-only params on all overridable methods, so overrides
    # are free to pick their own names.
    def __call__(self, span: Span, /) -> AsyncGenerator[None, Any] | None:
        """Return the generator frame for one span, or ``None``.

        The decorated class's ``__call__`` overrides this.
        """
        return None

    async def on_span_start(self, span: Span, /) -> None:
        gen = self(span)
        if gen is None:
            return
        try:
            await anext(gen)
        except StopAsyncIteration:
            return  # returned before yielding: opted out of this span
        if self.__live is None:
            self.__live = {}
        self.__live[span.id] = gen

    async def on_span_event(self, span: Span, event: SpanEvent, /) -> None:
        live = self.__live
        gen = live.get(span.id) if live is not None else None
        if live is None or gen is None:
            return  # opted out, span already ended, or start raised
        try:
            await gen.asend(event)
        except StopAsyncIteration:
            # Finished mid-span: opted out of the rest of this span,
            # including its end.
            del live[span.id]
        except BaseException:
            # The generator frame is dead, don't resume it again at
            # span end.  The error itself is logged by _dispatch.
            del live[span.id]
            raise

    async def on_span_end(self, span: Span, /) -> None:
        gen = self.__live.pop(span.id, None) if self.__live else None
        if gen is None:  # opted out, or start raised (and was logged)
            return
        try:
            await anext(gen)
        except StopAsyncIteration:
            return
        await gen.aclose()
        raise RuntimeError("adapter generator yielded again after span end")


class _AdapterFn(AdapterMixin):
    """Adapter built by :func:`adapter` from a function."""

    def __init__(self, fn: Callable[[Span], AsyncGenerator[None, Any]]) -> None:
        self._fn = fn

    def __repr__(self) -> str:
        return f"adapter({getattr(self._fn, '__qualname__', self._fn)!r})"

    def __call__(self, span: Span, /) -> AsyncGenerator[None, Any]:
        return self._fn(span)


_AdapterCallableT = TypeVar("_AdapterCallableT", bound=AdapterCallable)


@overload
def adapter(target: type[_AdapterCallableT]) -> type[_AdapterCallableT]: ...


@overload
def adapter(
    target: Callable[[Span], AsyncGenerator[None, Any]],
) -> AdapterMixin: ...


def adapter(
    target: type[_AdapterCallableT]
    | Callable[[Span], AsyncGenerator[None, Any]],
) -> type[_AdapterCallableT] | AdapterMixin:
    """Build an adapter from an async generator function or class.

    Decorating a free-standing async generator function returns a
    ready-to-register adapter.

    Decorating a class whose ``__call__`` is an
    async generator method returns that class with the adapter
    machinery (:class:`AdapterMixin`) mixed in::

        @adapter
        class Vendor:
            def __init__(self, *, api_key):
                self._client = sdk.Client(api_key)

            async def __call__(self, span):
                ...  # same yield loop as the function form

        ai.experimental_telemetry.register(Vendor(api_key="..."))

    How to use to bridge a vendor SDKs:
    write the vendor's ``with``/``async with`` around one yield loop.
    Code before the loop runs at span start; each span event resumes the
    yield with the :class:`SpanEvent`, live; span end resumes it with
    ``None``, and the code after the loop runs with ``span.data``
    fully populated and ``ended_at`` set::

        @adapter
        async def vendor(span):
            with sdk.start_span(span.name) as v:
                while (ev := (yield)) is not None:   # each event, live
                    v.log_event(ev.name, timestamp=ev.time_ns)
                v.update(output=span.data)           # span end

        ai.experimental_telemetry.register(vendor)

    - A failed span ends the loop normally, like any other; read
      ``span.error`` (a serializable :class:`SpanError`, never a live
      exception — the span may have failed in another process) after
      the loop to report it.
    - A span that lived elsewhere and arrives complete is replayed to
      the generator as start, events, end, back to back — write the
      bridge against the span's fields and it handles both live and
      after-the-fact delivery.
    - Returning before the first ``yield`` skips that span — cheap
      filtering by span type.  Returning mid-span, from inside the
      loop, opts out of the rest of that span, including its end.
    - Don't make a bare ``yield`` that ignores the sent value, because
      it runs the code after it on the first event, not at span
      end.  Always loop until the yield returns ``None``.
    - Like any adapter, a generator that raises is logged and skipped;
      it never kills the run.
    """
    if inspect.isclass(target):
        if issubclass(target, AdapterMixin):
            raise TypeError(f"{target.__name__} is already an adapter")
        call = inspect.getattr_static(target, "__call__", None)
        if not inspect.isasyncgenfunction(call):
            raise TypeError(
                "@adapter on a class requires __call__ to be an async "
                "generator method (`async def` containing a `yield` loop)"
            )
        mixed = type(
            target.__name__,
            (target, AdapterMixin),
            {
                "__module__": target.__module__,
                "__qualname__": target.__qualname__,
                "__doc__": target.__doc__,
            },
        )
        return cast("type[_AdapterCallableT]", mixed)
    if not inspect.isasyncgenfunction(target):
        raise TypeError(
            "@adapter requires an async generator function "
            "(`async def` containing a `yield` loop)"
        )
    return _AdapterFn(target)
