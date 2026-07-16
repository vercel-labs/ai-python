"""Telemetry: spans, sinks, adapters, and the ambient current span.

Experimental: not part of the stable API, may change or be removed.

:class:`Span` is a serializable record of work done by the application.
It carries ids, timestamps, parentage, and typed data — lifecycle is
data (``started_at``/``ended_at`` fields), not object state, so a span
can start in one process and end in another.

The one lifecycle verb is :meth:`Span.push`: set fields, then push the
updated span. A push delivers a frozen snapshot to the current *sink*.
By default that sink drives the adapter registry, so the plain case
works exactly like classic start/end callbacks::

    async with ai.experimental_telemetry.span("retrieval", query=q) as sp:
        docs = await search(q)
        sp.set(count=len(docs))

Nesting is automatic: the current span is tracked using a context var.

The lower-level api the context manager is built on::

    sp = create_span("turn", session=sid)   # identity only, reports nothing
    sp.started_at = now_ns()
    await sp.push()                          # visible to sinks/adapters
    ...                                      # possibly elsewhere, later:
    sp.ended_at = now_ns()
    await sp.push()                          # complete

Because ``Span`` is a pydantic model it crosses process boundaries like
any other data (``model_dump`` / ``model_validate``); restore it and
keep going — open children under it with ``parent=sp`` or
:func:`use_span`, or finish it and push.

An adapter processes spans and decides what to do with them::

    class MyAdapter:
        # all optional and can be blocking
        async def on_span_start(self, span): ...
        async def on_span_end(self, span): ...
        async def on_span_event(self, span, event): ...

    ai.experimental_telemetry.register(MyAdapter())

Adapters dispatch on the type of ``span.data``.  An adapter that crashes
is logged and skipped, it never kills the run.

A :class:`Sink` receives raw pushed snapshots instead. :func:`use_sink`
reroutes pushes within a context — e.g. a :class:`Collector` inside a
durable workflow body gathers spans as data so a step can re-push them
to the real adapters exactly once.

A :class:`SpanEvent` is a named, timestamped milestone inside a span's
lifetime (``first_token``, ``hook_resolved``, ...): append it to
``span.events`` and push.

Span timestamps come from the ambient clock (:func:`now_ns`);
:func:`use_clock` overrides it per-context, e.g. with a deterministic
clock inside a durable workflow.

For bridging vendor SDKs without a class, :func:`wrap_span` builds an
adapter from a free-standing async generator function.
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
    overload,
)

import pydantic

# ``typing.TypeVar`` lacks the ``default=`` kwarg on Python <3.13.
# Use the typing_extensions backport so this works on 3.12 too.
from typing_extensions import TypeVar

from .. import util
from ..types import messages as messages_
from ..types import usage as usage_

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Iterator,
    )

    from ..models.core import params as params_

    # Real types for checkers; ``Any`` at runtime.  The params module
    # lives under ``ai.models``, which imports back into this module —
    # and rehydrating them wouldn't be faithful anyway
    # (``provider_params`` is keyed by type objects, which don't
    # round-trip through JSON).
    _InferenceParams = params_.InferenceRequestParams
    _GenerateParams = params_.GenerateParams
else:
    _InferenceParams = Any
    _GenerateParams = Any

logger = logging.getLogger(__name__)


# ── Clock ─────────────────────────────────────────────────────────

_span_clock: contextvars.ContextVar[Callable[[], int] | None] = (
    contextvars.ContextVar("span_clock", default=None)
)


def now_ns() -> int:
    """Nanoseconds since the epoch, from the ambient span clock.

    The wall clock by default; whatever :func:`use_clock` installed
    otherwise.  Use it to stamp ``started_at``/``ended_at`` and span
    events.
    """
    clock = _span_clock.get()
    return clock() if clock is not None else time.time_ns()


@util.contextmanager_any_sync
def use_clock(now_ns: Callable[[], int]) -> Iterator[None]:
    """Read span timestamps from ``now_ns`` within this context.

    Framework's observability creates timestamps. This API can be
    used to plug an approved clock function in durable execution
    settings::

        with ai.experimental_telemetry.use_clock(workflow.time_ns):
            ...  # spans opened here read time from workflow.time_ns

    This can also be used as a decorator on both sync and async
    functions::

        @ai.experimental_telemetry.use_clock(clock.time_ns)
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

    Implement it with a pydantic model to make your own typed spans —
    no base class, no registration::

        class RetrievalSpanData(pydantic.BaseModel):
            kind: Literal["retrieval"] = "retrieval"
            query: str
            count: int | None = None

        data = RetrievalSpanData(query=q)
        async with ai.experimental_telemetry.span(data) as sp:
            docs = await search(q)
            sp.data.count = len(docs)  # typed

    ``kind`` says what kind of work the span records: it doubles as
    the span's default name and travels with the serialized data.  On
    a bare ``Span.model_validate`` a user-defined type stays a plain
    dict (only the framework's own kinds are rebuilt); restore with
    ``Span[RetrievalSpanData].model_validate(...)`` to get it typed.

    Adapters match on it the same way they match on framework types:
    ``case RetrievalSpanData() as d: ...``.
    """

    @property
    def kind(self) -> str: ...


class RunSpanData(pydantic.BaseModel):
    """One ``Agent.run``: the whole loop.

    ``blocked``/``final_message`` are set at span end: ``blocked`` is
    True when the run ended suspended on an unresolved hook (see
    ``AgentStream.blocked``), ``final_message`` is the last assistant
    message produced, if any.
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


class LoopTurnSpanData(pydantic.BaseModel):
    """One turn of the default agent loop.

    Carries no fields — it exists so adapters can group a turn's model
    and tool spans.  Turn order is given by ``started_at``/``parent_id``.
    """

    kind: Literal["loop_turn"] = "loop_turn"


class AiStreamSpanData(pydantic.BaseModel):
    """One streaming LLM call.  ``message``/``usage`` are set at span end."""

    kind: Literal["ai_stream"] = "ai_stream"
    model: str
    messages: list[messages_.Message]
    params: _InferenceParams | None = None
    provider: str | None = None
    tool_names: list[str] | None = None
    message: messages_.Message | None = None
    usage: usage_.Usage | None = None


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
    """

    kind: Literal["tool_execution"] = "tool_execution"
    tool_name: str
    tool_call_id: str
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
    """A user span made with ``span("name", key=value, ...)``."""

    kind: Literal["custom"] = "custom"
    attributes: dict[str, Any]


# names for span events shared between producers,
# adapters, and tests
FIRST_TOKEN = "first_token"
RESPONSE_COMPLETE = "response_complete"
HOOK_DEFERRED = "hook_deferred"
HOOK_RESOLVED = "hook_resolved"
HOOK_CANCELLED = "hook_cancelled"


class SpanEvent(pydantic.BaseModel):
    """A named, timestamped milestone inside a span's lifetime.

    Append to ``span.events`` and push; the next push delivers it::

        sp.events.append(SpanEvent(name=FIRST_TOKEN, time_ns=now_ns(),
                                   attributes={}))
        await sp.push()

    Ordering: events on one span are delivered in list order; there is
    no ordering guarantee across spans.
    """

    name: str
    time_ns: int
    attributes: dict[str, Any]


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


# The serialized ``kind`` field identifies framework span data, so
# pydantic rebuilds the typed form natively when a dumped span is
# validated (the union discriminates on it).
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

# Covariant: adapters take ``Span`` (= ``Span[SpanData]``) and must
# accept any concretely-typed span.  They read ``data``, never replace
# it, so the widened reference is safe in practice.
#
# The runtime default is the framework union with an ``Any`` fallback:
# a bare ``Span`` rebuilds framework data by its ``kind`` tag and
# passes everything else through untouched — a live user-typed data
# instance stays itself, an unrecognized dumped dict stays a dict (a
# restored span must never fail on its data).  A parametrized
# ``Span[MyData]`` rebuilds the user type instead.  The bare schema is
# built here, at import: sandboxed callers (durable workflow bodies)
# can validate but not build.
if TYPE_CHECKING:
    DataT_co = TypeVar(
        "DataT_co", bound=SpanData, default=SpanData, covariant=True
    )
else:
    DataT_co = TypeVar("DataT_co", default=_FrameworkData | Any)
# Function-scoped variant for ``span()``: a covariant variable can't
# appear as a parameter.
DataT = TypeVar("DataT", bound=SpanData)


class Span(pydantic.BaseModel, Generic[DataT_co]):
    """A serializable record of a unit of work.

    Generic in its data type: ``span(RetrievalSpanData(...))`` gives a
    ``Span[RetrievalSpanData]``, so late assignments to ``sp.data``
    fields are type checked.  A bare ``Span`` is ``Span[SpanData]``.

    Lifecycle is carried by the timestamp fields: a span with
    ``started_at=None`` hasn't started; one with ``ended_at`` set is
    complete.  Mutate fields, then :meth:`push` to report the new
    state.  Nothing is reported except by pushing.

    A span round-trips through ``model_dump``/``model_validate`` like
    any pydantic model.  Restoring rebuilds typed ``data`` for the
    framework's span types, matched by the ``kind`` tag serialized
    inside the data itself; data of user-defined span types stays a
    plain dict unless restored with ``Span[MyData].model_validate(...)``.
    Adapters receiving re-pushed spans should treat dict data as the
    fallback it is.

    ``replay=True`` marks work that is being replayed (resume,
    serverless re-entry) rather than performed live.

    ``set_as_current=False`` marks a span that does not set itself
    as a current. This is used by ai.stream's span that stays open for
    the duration of a loop turn because of the context manager api.
    Adapters must not make those spans "current" either to avoid
    nesting discrepancies.

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

    schema_version: ClassVar[int] = 3

    def set(self, **attributes: Any) -> None:
        """Attach attributes to a span created with ``span("name", ...)``."""
        if not isinstance(self.data, CustomSpanData):
            raise TypeError(
                "set() only works on user spans; framework spans carry "
                "typed data — assign its fields directly"
            )
        self.data.attributes.update(attributes)

    async def push(self) -> None:
        """Deliver a frozen copy of this span to the current sink.

        The default sink drives the registered adapters (see
        :func:`register`); :func:`use_sink` reroutes pushes within a
        context.  Each push snapshots the whole span, so keep mutating
        and push again as the work progresses — the last push with
        ``ended_at`` set is the complete record.

        Telemetry never kills the run: a sink (or adapter) failure is
        logged and swallowed.
        """
        sink = _current_sink.get() or _registry_sink
        try:
            await sink.emit(self.model_copy(deep=True))
        except Exception:
            logger.exception("telemetry sink %r raised in emit", sink)


# ── Current span ──────────────────────────────────────────────────

_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)


def current() -> Span | None:
    """Return the current span, or ``None`` when no span is open."""
    return _current.get()


@util.contextmanager_any_sync
def use_span(span_: Span) -> Iterator[None]:
    """Make ``span_`` the current span within this context.

    Unlike :func:`span` this is pure context plumbing — no timestamps,
    no pushes.  Use it to continue a trace around existing work, e.g.
    parenting under a span restored from another process::

        turn_span = Span.model_validate(payload["turn_span"])
        with ai.experimental_telemetry.use_span(turn_span):
            ...  # spans opened here parent under turn_span
    """
    token = _current.set(span_)
    try:
        yield
    finally:
        _current.reset(token)


# ── Sinks ─────────────────────────────────────────────────────────


class Sink(Protocol):
    """Anything that accepts pushed span snapshots."""

    async def emit(self, span_: Span, /) -> None: ...


_current_sink: contextvars.ContextVar[Sink | None] = contextvars.ContextVar(
    "current_sink", default=None
)


@util.contextmanager_any_sync
def use_sink(sink: Sink) -> Iterator[None]:
    """Route span pushes to ``sink`` within this context.

    The default (outside any ``use_sink``) is the adapter registry.
    Inside a durable workflow body, route to a :class:`Collector`
    instead and re-push the collected spans from a step — the body
    re-runs on every replay, the step doesn't.
    """
    token = _current_sink.set(sink)
    try:
        yield
    finally:
        _current_sink.reset(token)


class Collector:
    """A sink that keeps the latest snapshot of every span pushed to it.

    ``spans`` maps span id to the most recent snapshot, in first-push
    order.  Scoop them out as data (``model_dump``) and re-push them
    where the real sink is available::

        collector = Collector()
        with use_sink(collector):
            ...  # replayed / suspendable code
        payload = [s.model_dump(mode="json") for s in collector.spans.values()]

        # elsewhere (a workflow step, another process):
        for data in payload:
            await Span.model_validate(data).push()
    """

    def __init__(self) -> None:
        self.spans: dict[str, Span] = {}

    async def emit(self, span_: Span, /) -> None:
        self.spans[span_.id] = span_


class _RegistrySink:
    """The default sink: drives registered adapters.

    Translates a stream of span snapshots into the classic
    ``on_span_start`` / ``on_span_event`` / ``on_span_end`` callbacks.
    One live *view* object is kept per span id and updated in place on
    every push, so an adapter holding the span between callbacks reads
    current data at span end.

    Rules:

    - a snapshot with ``started_at=None`` reports nothing (the span
      hasn't started);
    - the first snapshot of an id fires ``on_span_start``, then
      ``on_span_event`` per event, then ``on_span_end`` if the snapshot
      is already complete — so a span that lived elsewhere and arrives
      finished is delivered whole;
    - later snapshots fire ``on_span_event`` for events not seen
      before, and ``on_span_end`` once ``ended_at`` appears;
    - after the end the id is forgotten: pushing a completed span again
      re-delivers it in full (the durable re-emission path — dedup, if
      any, belongs to the backend, keyed on the span id).
    """

    def __init__(self) -> None:
        self._views: dict[str, Span] = {}

    async def emit(self, span_: Span, /) -> None:
        if span_.started_at is None:
            return
        view = self._views.get(span_.id)
        if view is None:
            view = span_
            self._views[view.id] = view
            await _dispatch("on_span_start", view)
            fresh = list(view.events)
        else:
            seen = len(view.events)
            view.__dict__.update(span_.__dict__)
            fresh = view.events[seen:]
        for event in fresh:
            await _dispatch("on_span_event", view, event)
        if view.ended_at is not None:
            del self._views[view.id]
            await _dispatch("on_span_end", view)


_registry_sink = _RegistrySink()


# ── Adapter registry ──────────────────────────────────────────────

_adapters: list[Any] = []


def register(adapter: Any) -> None:
    """Add an adapter.  Multiple adapters coexist independently."""
    _adapters.append(adapter)


def unregister(adapter: Any) -> None:
    """Remove a previously registered adapter."""
    _adapters.remove(adapter)


async def _dispatch(method: str, span_: Span, *args: Any) -> None:
    for adapter in list(_adapters):
        fn = getattr(adapter, method, None)
        if fn is None:
            continue
        try:
            result = fn(span_, *args)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "telemetry adapter %r raised in %s", adapter, method
            )


async def flush() -> None:
    """Flush the current sink and every adapter that supports it.

    Vendor SDKs buffer spans in background threads; call this before a
    checkpoint or process exit so everything pushed so far is actually
    delivered.  Sinks and adapters opt in by defining ``flush()``
    (sync or async); failures are logged and skipped.
    """
    sink = _current_sink.get()
    targets = ([sink] if sink is not None else []) + list(_adapters)
    for target in targets:
        fn = getattr(target, "flush", None)
        if fn is None:
            continue
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("telemetry flush of %r raised", target)


# ── Creating spans ────────────────────────────────────────────────


@overload
def create_span(
    name_or_data: str,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
    **attributes: Any,
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
    **attributes: Any,
) -> Span[Any]:
    """Create a span: identity only, nothing is reported.

    Mints the span id, and takes trace id and parentage from ``parent``
    (default: the current span; a fresh trace when there is none).
    ``parent`` may be a span restored from another process — that is
    how a trace continues across a boundary.

    The span has no timestamps yet: stamp ``started_at`` (see
    :func:`now_ns`) and :meth:`Span.push` when the work begins, or
    hand the whole lifecycle to the :func:`span` context manager.
    """
    if isinstance(name_or_data, str):
        name = name_or_data
        data: SpanData = CustomSpanData(attributes=dict(attributes))
    else:
        if attributes:
            raise TypeError("attributes only go with a str span name")
        name = name_or_data.kind
        data = name_or_data
    if parent is None:
        parent = _current.get()
    if parent is None:
        trace_id, parent_id = messages_.generate_id("trace"), None
    else:
        trace_id, parent_id = parent.trace_id, parent.id
    return Span(
        name=name,
        data=data,
        id=messages_.generate_id("span"),
        trace_id=trace_id,
        parent_id=parent_id,
        replay=replay,
        set_as_current=set_as_current,
    )


@overload
def span(
    name_or_data: str,
    /,
    *,
    parent: Span | None = None,
    replay: bool = False,
    set_as_current: bool = True,
    **attributes: Any,
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
    **attributes: Any,
) -> contextlib.AbstractAsyncContextManager[Span[Any]]:
    """Open a span; it is "current" (parents new spans) inside the block.

    Sugar over the data api: creates the span, stamps ``started_at``
    and pushes on enter; stamps ``ended_at`` (and ``error``, if the
    block raised) and pushes on exit.

    Pass a name plus attributes for a user span, or a :class:`SpanData`
    instance for a typed one — the span is generic in it, so late
    assignments to ``sp.data`` fields are type checked.  Exceptions are
    recorded on the span and re-raised.

    ``parent`` overrides the ambient parent for this span: a live
    :class:`Span`, or one restored from another process to continue its
    trace here.  The default parents under the current span.

    ``set_as_current=False`` keeps the span from becoming current:
    work done while it is open parents to *its* parent instead. Used by
    ai.stream because of the context manager api.
    """
    # The indirection exists because type checkers can't apply
    # ``asynccontextmanager`` to an overloaded function directly.
    return _span_impl(
        name_or_data,
        parent=parent,
        replay=replay,
        set_as_current=set_as_current,
        **attributes,
    )


@contextlib.asynccontextmanager
async def _span_impl(
    name_or_data: str | SpanData,
    /,
    *,
    parent: Span | None,
    replay: bool,
    set_as_current: bool,
    **attributes: Any,
) -> AsyncIterator[Span[Any]]:
    sp = create_span(
        name_or_data,
        parent=parent,
        replay=replay,
        set_as_current=set_as_current,
        **attributes,
    )
    sp.started_at = now_ns()
    await sp.push()
    token = _current.set(sp) if set_as_current else None
    try:
        yield sp
    except BaseException as exc:
        # GeneratorExit is how a consumer closes a stream early —
        # normal control flow, not a failure of the spanned work.
        if not isinstance(exc, GeneratorExit) and sp.error is None:
            sp.error = SpanError.from_exception(exc)
        raise
    finally:
        # A span that set itself as current must be closed while it is
        # still current, in the task that opened it.  Otherwise
        # resetting the context token would silently corrupt the
        # current-span state (new spans would parent under an already-
        # closed span) — end the span for adapters, then raise loudly.
        misordered = False
        if token is not None:
            if _current.get() is not sp:
                misordered = True
            else:
                try:
                    _current.reset(token)
                except ValueError:
                    # Token from a different task's context.
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


class Adapter:
    """Base class for adapters: the protocol, with the right defaults.

    class Vendor(telemetry.Adapter):
        async def wrap_span(self, span):
            with sdk.start_span(span.name) as v:
                while (ev := (yield)) is not None:
                    v.log_event(ev.name)
                if span.error is not None:
                    v.set_error(span.error.message)
                v.update(output=span.data)
    """

    # Name-mangled so the driver's bookkeeping can never collide with
    # a subclass attribute; created lazily so subclasses keep their
    # ``__init__`` entirely to themselves.
    __live: dict[str, AsyncGenerator[None, Any]] | None = None

    # Positional-only params on all overridable methods, so subclasses
    # are free to pick their own names (`span` shadows nothing there).
    def wrap_span(self, span_: Span, /) -> AsyncGenerator[None, Any] | None:
        """Return the generator frame for one span, or ``None``.

        Override it as an async generator method (a ``yield`` loop, as
        in the class docstring) — calling one *returns* the generator,
        so the override satisfies this signature.  The default returns
        ``None``: no per-span frame.
        """
        return None

    async def on_span_start(self, span_: Span, /) -> None:
        gen = self.wrap_span(span_)
        if gen is None:
            return
        try:
            await anext(gen)
        except StopAsyncIteration:
            return  # returned before yielding: opted out of this span
        if self.__live is None:
            self.__live = {}
        self.__live[span_.id] = gen

    async def on_span_event(self, span_: Span, event: SpanEvent, /) -> None:
        live = self.__live
        gen = live.get(span_.id) if live is not None else None
        if live is None or gen is None:
            return  # opted out, span already ended, or start raised
        try:
            await gen.asend(event)
        except StopAsyncIteration:
            # Finished mid-span: opted out of the rest of this span,
            # including its end.
            del live[span_.id]
        except BaseException:
            # The generator frame is dead, don't resume it again at
            # span end.  The error itself is logged by _dispatch.
            del live[span_.id]
            raise

    async def on_span_end(self, span_: Span, /) -> None:
        gen = self.__live.pop(span_.id, None) if self.__live else None
        if gen is None:  # opted out, or start raised (and was logged)
            return
        try:
            await anext(gen)
        except StopAsyncIteration:
            return
        await gen.aclose()
        raise RuntimeError("wrap_span generator yielded again after span end")


class _WrapSpanFn(Adapter):
    """Adapter built by :func:`wrap_span`."""

    def __init__(self, fn: Callable[[Span], AsyncGenerator[None, Any]]) -> None:
        self._fn = fn

    def __repr__(self) -> str:
        return f"wrap_span({getattr(self._fn, '__qualname__', self._fn)!r})"

    def wrap_span(self, span_: Span, /) -> AsyncGenerator[None, Any]:
        return self._fn(span_)


def wrap_span(
    fn: Callable[[Span], AsyncGenerator[None, Any]],
) -> Adapter:
    """Build an adapter from a free-standing async generator function.

    Sugar over subclassing :class:`Adapter` — same semantics as
    overriding its ``wrap_span`` method, for bridges that need no
    state or methods of their own.

    How to use to bridge a vendor SDKs:
    write the vendor's ``with``/``async with`` around one yield loop.
    Code before the loop runs at span start; each span event resumes the
    yield with the :class:`SpanEvent`, live; span end resumes it with
    ``None``, and the code after the loop runs with ``span.data``
    fully populated and ``ended_at`` set::

        @wrap_span
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
    if not inspect.isasyncgenfunction(fn):
        raise TypeError(
            "wrap_span requires an async generator function "
            "(`async def` containing a `yield` loop)"
        )
    return _WrapSpanFn(fn)
