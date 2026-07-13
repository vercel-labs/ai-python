"""Telemetry: spans, adapters, and the ambient current span.

:class:`Span` is a record of work done by the application. It carries
start, end, parent, and typed data.

Same API is used to instrument the framework and define custom spans::

    async with ai.telemetry.span("retrieval", query=q) as sp:
        docs = await search(q)
        sp.set(count=len(docs))

Nesting is automatic: the current span is tracked using a context var.

An adapter processes spans and decides what to do with them::

    class MyAdapter:
        # all optional and can be blocking
        async def on_span_start(self, span): ...
        async def on_span_end(self, span): ...
        async def on_span_event(self, span, event): ...

    ai.telemetry.register(MyAdapter())

Adapters dispatch on the type of ``span.data``.  An adapter that crashes
is logged and skipped, it never kills the run.

A :class:`SpanEvent` is a named, timestamped milestone inside a span's
lifetime (``first_token``, ``hook_resolved``, ...), recorded with
:meth:`Span.add_event`.

Span timestamps come from the ambient clock; :func:`use_clock`
overrides it per-context, e.g. with a deterministic clock inside a
durable workflow.

For bridging vendor SDKs without a class, :func:`wrap_span` builds an
adapter from a free-standing async generator function.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import inspect
import logging
import time
from typing import (
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Iterator,
    )

    from ..models.core import params as params_
    from ..types import usage as usage_

logger = logging.getLogger(__name__)


# ── Clock ─────────────────────────────────────────────────────────

_span_clock: contextvars.ContextVar[Callable[[], int] | None] = (
    contextvars.ContextVar("span_clock", default=None)
)


def _now_ns() -> int:
    clock = _span_clock.get()
    return clock() if clock is not None else time.time_ns()


@util.contextmanager_any_sync
def use_clock(now_ns: Callable[[], int]) -> Iterator[None]:
    """Read span timestamps from ``now_ns`` within this context.

    Framework's observability creates timestamps. This API can be
    used to plug an approved clock function in durable execution
    settings::

        with ai.telemetry.use_clock(workflow.time_ns):
            ...  # spans opened here read time from workflow.time_ns

    This can also be used as a decorator on both sync and async
    functions::

        @ai.telemetry.use_clock(clock.time_ns)
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
    """Anything with a ``span_name`` is span data.

    Implement it with a plain dataclass to make your own typed spans —
    no base class, no registration::

        @dataclasses.dataclass
        class RetrievalSpanData:
            query: str
            count: int | None = None

            span_name: ClassVar[str] = "retrieval"

        async with ai.telemetry.span(RetrievalSpanData(query=q)) as sp:
            docs = await search(q)
            sp.data.count = len(docs)  # typed

    Adapters match on it the same way they match on framework types:
    ``case RetrievalSpanData() as d: ...``.
    """

    span_name: ClassVar[str]


# Covariant: adapters take ``Span`` (= ``Span[SpanData]``) and must
# accept any concretely-typed span.  They read ``data``, never replace
# it, so the widened reference is safe in practice.
DataT_co = TypeVar("DataT_co", bound=SpanData, default=SpanData, covariant=True)
# Function-scoped variant for ``span()``: a covariant variable can't
# appear as a parameter.
DataT = TypeVar("DataT", bound=SpanData)


@dataclasses.dataclass
class RunSpanData:
    """One ``Agent.run``: the whole loop."""

    agent: str
    model: str
    messages: list[messages_.Message]

    span_name: ClassVar[str] = "run"


@dataclasses.dataclass
class LoopTurnSpanData:
    """One turn of the default agent loop.

    Carries no fields — it exists so adapters can group a turn's model
    and tool spans.  Turn order is given by ``started_at``/``parent_id``.
    """

    span_name: ClassVar[str] = "loop_turn"


@dataclasses.dataclass
class AiStreamSpanData:
    """One streaming LLM call.  ``message``/``usage`` are set at span end."""

    model: str
    messages: list[messages_.Message]
    params: params_.InferenceRequestParams | None = None
    message: messages_.Message | None = None
    usage: usage_.Usage | None = None

    span_name: ClassVar[str] = "ai_stream"


@dataclasses.dataclass
class AiGenerateSpanData:
    """One non-streaming generation call (images, video, ...)."""

    model: str
    messages: list[messages_.Message]
    params: params_.GenerateParams | None = None
    message: messages_.Message | None = None

    span_name: ClassVar[str] = "ai_generate"


@dataclasses.dataclass
class ToolExecutionSpanData:
    """One tool execution, from dispatch to result."""

    tool_name: str
    tool_call_id: str
    args: dict[str, Any] | None = None
    result: Any = None
    is_error: bool = False

    span_name: ClassVar[str] = "tool_execution"


@dataclasses.dataclass
class HookSpanData:
    """One hook suspension, from deferred until resolved or cancelled."""

    label: str
    hook_type: str
    metadata: dict[str, Any]
    status: Literal["pending", "resolved", "cancelled"] = "pending"

    span_name: ClassVar[str] = "hook"


@dataclasses.dataclass
class CustomSpanData:
    """A user span made with ``span("name", key=value, ...)``."""

    attributes: dict[str, Any]

    span_name: ClassVar[str] = "custom"


# names for span events shared between producers,
# adapters, and tests
FIRST_TOKEN = "first_token"
RESPONSE_COMPLETE = "response_complete"
HOOK_DEFERRED = "hook_deferred"
HOOK_RESOLVED = "hook_resolved"
HOOK_CANCELLED = "hook_cancelled"


@dataclasses.dataclass
class SpanEvent:
    """A named, timestamped milestone inside a span's lifetime."""

    name: str
    time_ns: int
    attributes: dict[str, Any]


class SpanRef(pydantic.BaseModel):
    """A serializable reference to a span: just enough to parent under it.

    Capture with :func:`current_ref` or ``span.ref``; carry it across a
    checkpoint, a queue, or the network like any pydantic model
    (``model_dump`` / ``model_validate``); restore by opening a span
    with ``parent=ref`` on the other side::

        ref = ai.telemetry.current_ref()
        job = {"task": task, "telemetry": ref.model_dump()}

        # elsewhere:
        ref = ai.telemetry.SpanRef.model_validate(job["telemetry"])
        async with ai.telemetry.span("pickup", parent=ref):
            ...  # everything inside continues the original trace

    ``sampled`` is carried so refs round-trip sampling decisions
    losslessly; the framework itself does not act on it yet.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    trace_id: str
    span_id: str
    sampled: bool = True


@dataclasses.dataclass
class Span(Generic[DataT_co]):
    """A record of a unit of work.

    Generic in its data type: ``span(RetrievalSpanData(...))`` gives a
    ``Span[RetrievalSpanData]``, so late assignments to ``sp.data``
    fields are type checked.  A bare ``Span`` is ``Span[SpanData]``.

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
    parent_id: str | None
    started_at: int  # nanoseconds since the epoch, from the ambient clock
    ended_at: int | None = None
    error: BaseException | None = None
    replay: bool = False
    set_as_current: bool = True
    span_events: list[SpanEvent] = dataclasses.field(default_factory=list)

    schema_version: ClassVar[int] = 1

    @property
    def ref(self) -> SpanRef:
        """A serializable :class:`SpanRef` pointing at this span."""
        return SpanRef(trace_id=self.trace_id, span_id=self.id)

    def set(self, **attributes: Any) -> None:
        """Attach attributes to a span created with ``span("name", ...)``."""
        if not isinstance(self.data, CustomSpanData):
            raise TypeError(
                "set() only works on user spans; framework spans carry "
                "typed data — assign its fields directly"
            )
        self.data.attributes.update(attributes)

    async def add_event(self, name: str, **attributes: Any) -> SpanEvent:
        """Record a milestone on this span and dispatch it to adapters.

        Stamps the current time, appends to ``span_events``, and
        dispatches ``on_span_event(span, event)`` to every adapter with
        the usual isolation (a raising adapter is logged and skipped).

        Ordering: events on one span are appended in wall-clock order
        (the order ``add_event`` was called in); there is no ordering
        guarantee across spans.

        Adding an event to a span that already ended logs a warning
        but still records and dispatches the event, a late milestone
        is better reported late than dropped.
        """
        event = SpanEvent(
            name=name, time_ns=_now_ns(), attributes=dict(attributes)
        )
        if self.ended_at is not None:
            logger.warning(
                "span event %r added to already-ended span %r", name, self.name
            )
        self.span_events.append(event)
        await _dispatch("on_span_event", self, event)
        return event


# ── Current span + adapter registry ───────────────────────────────

_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)

_adapters: list[Any] = []


def current() -> Span | None:
    """Return the current span, or ``None`` when no span is open."""
    return _current.get()


def current_ref() -> SpanRef | None:
    """Return a :class:`SpanRef` to the current span, ``None`` outside one.

    This is the capture side of crossing a process boundary: checkpoint
    or send the ref, then open a span with ``parent=ref`` on the other
    side.
    """
    sp = _current.get()
    return sp.ref if sp is not None else None


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


@overload
def span(
    name_or_data: str,
    /,
    *,
    parent: Span | SpanRef | None = None,
    replay: bool = False,
    set_as_current: bool = True,
    **attributes: Any,
) -> contextlib.AbstractAsyncContextManager[Span[CustomSpanData]]: ...


@overload
def span(
    name_or_data: DataT,
    /,
    *,
    parent: Span | SpanRef | None = None,
    replay: bool = False,
    set_as_current: bool = True,
) -> contextlib.AbstractAsyncContextManager[Span[DataT]]: ...


def span(
    name_or_data: str | SpanData,
    /,
    *,
    parent: Span | SpanRef | None = None,
    replay: bool = False,
    set_as_current: bool = True,
    **attributes: Any,
) -> contextlib.AbstractAsyncContextManager[Span[Any]]:
    """Open a span; it is "current" (parents new spans) inside the block.

    Pass a name plus attributes for a user span, or a :class:`SpanData`
    instance for a typed one — the span is generic in it, so late
    assignments to ``sp.data`` fields are type checked.  Exceptions are
    recorded on the span and re-raised.

    ``parent`` overrides the ambient parent for this span: a live
    :class:`Span`, or a :class:`SpanRef` restored from another process
    to continue its trace here.  The default parents under the current
    span.

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
    parent: Span | SpanRef | None,
    replay: bool,
    set_as_current: bool,
    **attributes: Any,
) -> AsyncIterator[Span[Any]]:
    if isinstance(name_or_data, str):
        name = name_or_data
        data: SpanData = CustomSpanData(attributes=dict(attributes))
    else:
        if attributes:
            raise TypeError("attributes only go with a str span name")
        name = name_or_data.span_name
        data = name_or_data
    if parent is None:
        parent = _current.get()
    parent_id: str | None
    match parent:
        case Span():
            trace_id, parent_id = parent.trace_id, parent.id
        case SpanRef():
            trace_id, parent_id = parent.trace_id, parent.span_id
        case None:
            trace_id, parent_id = messages_.generate_id("trace"), None
    sp = Span(
        name=name,
        data=data,
        id=messages_.generate_id("span"),
        trace_id=trace_id,
        parent_id=parent_id,
        started_at=_now_ns(),
        replay=replay,
        set_as_current=set_as_current,
    )
    await _dispatch("on_span_start", sp)
    token = _current.set(sp) if set_as_current else None
    try:
        yield sp
    except BaseException as exc:
        # GeneratorExit is how a consumer closes a stream early —
        # normal control flow, not a failure of the spanned work.
        if not isinstance(exc, GeneratorExit):
            sp.error = exc
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
        sp.ended_at = _now_ns()
        await _dispatch("on_span_end", sp)
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
            if span_.error is not None:
                # Throw the span's error into the generator at its
                # yield, so a vendor context manager around the yield
                # records the failure exactly as if it wrapped the
                # work itself.
                await gen.athrow(span_.error)
            else:
                await anext(gen)
        except StopAsyncIteration:
            return
        except BaseException as exc:
            if exc is span_.error:
                return  # the thrown error propagated back out: expected
            raise  # the generator itself failed: logged by _dispatch
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

        ai.telemetry.register(vendor)

    - A span that ends with an error is thrown into the generator at
      the ``yield``, so the vendor context manager sees the failure.
      Use ``try/finally`` around the loop for code that must run on
      both paths; catching the error only suppresses it here, never
      for the application.
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
