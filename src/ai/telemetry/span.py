"""Telemetry: spans, adapters, and the ambient current span.

One concept: the :class:`Span` — something with a start, an end, a
parent, and typed data.  The framework instruments itself with the same
API users use for their own spans::

    async with ai.telemetry.span("retrieval", query=q) as sp:
        docs = await search(q)
        sp.set(count=len(docs))

Nesting is automatic: the span becomes "current" for the duration of the
block (a context variable), so anything opened inside — by the user or
by the framework — becomes a child.  This works across tasks because
task creation copies context.

An adapter is something that receives spans.  Two methods, both
optional, each may be sync or async::

    class MyAdapter:
        def on_span_start(self, span): ...   # live view of long runs
        def on_span_end(self, span): ...     # the main one

    ai.telemetry.register(MyAdapter())

Adapters dispatch on the type of ``span.data``.  An adapter that raises
is logged and skipped — it never kills the run.

For bridging context-manager-shaped vendor SDKs, :func:`wrap_span`
builds an adapter from an async generator function that yields once.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from ..types import messages as messages_

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from ..models.core import params as params_
    from ..types import usage as usage_

logger = logging.getLogger(__name__)


# ── Span data types ───────────────────────────────────────────────
#
# The data type tells you what kind of span it is; fields are the real
# framework objects (Message, Usage, ...), not pre-flattened strings.
# New features add new data types, never new adapter methods.


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


@dataclasses.dataclass
class RunSpanData:
    """One ``Agent.run``: the whole loop, turns nested underneath."""

    agent: str
    model: str
    messages: list[messages_.Message]

    span_name: ClassVar[str] = "run"


@dataclasses.dataclass
class LoopTurnSpanData:
    """One turn of the default agent loop."""

    index: int

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
    """One hook suspension, from pending until resolved or cancelled."""

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


# ── The span ──────────────────────────────────────────────────────


@dataclasses.dataclass
class Span:
    """A unit of work: a start, an end, a parent, and typed data.

    ``replay=True`` marks work that is being replayed (resume,
    serverless re-entry) rather than performed live — adapters can and
    should render it differently.

    ``set_as_current=False`` marks a span that does not parent the work
    done while it is open (see :func:`span`).  Adapters that bridge to
    an ambient-context SDK (otel, ...) must not make such spans
    "current" there either, or spans opened meanwhile mis-parent under
    them on the other side.

    ``schema_version`` tracks the shape of spans and their data types;
    it is bumped on breaking changes so adapters can detect them.
    """

    name: str
    data: SpanData
    id: str
    trace_id: str
    parent_id: str | None
    started_at: int  # time.time_ns()
    ended_at: int | None = None
    error: BaseException | None = None
    replay: bool = False
    set_as_current: bool = True

    schema_version: ClassVar[int] = 1

    def set(self, **attributes: Any) -> None:
        """Attach attributes to a span created with ``span("name", ...)``."""
        if not isinstance(self.data, CustomSpanData):
            raise TypeError(
                "set() only works on user spans; framework spans carry "
                "typed data — assign its fields directly"
            )
        self.data.attributes.update(attributes)


# ── Current span + adapter registry ───────────────────────────────

_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)

_adapters: list[Any] = []


def current() -> Span | None:
    """Return the current span, or ``None`` when no span is open."""
    return _current.get()


def register(adapter: Any) -> None:
    """Add an adapter.  Multiple adapters coexist independently."""
    _adapters.append(adapter)


def unregister(adapter: Any) -> None:
    """Remove a previously registered adapter."""
    _adapters.remove(adapter)


async def _dispatch(method: str, span_: Span) -> None:
    for adapter in list(_adapters):
        fn = getattr(adapter, method, None)
        if fn is None:
            continue
        try:
            result = fn(span_)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "telemetry adapter %r raised in %s", adapter, method
            )


@contextlib.asynccontextmanager
async def span(
    name_or_data: str | SpanData,
    /,
    *,
    replay: bool = False,
    set_as_current: bool = True,
    **attributes: Any,
) -> AsyncIterator[Span]:
    """Open a span; it is "current" (parents new spans) inside the block.

    Pass a name plus attributes for a user span, or a :class:`SpanData`
    instance for a typed one.  Exceptions are recorded on the span and
    re-raised.

    ``set_as_current=False`` keeps the span from becoming current: it
    still parents to the current span, but work done while it is open
    parents to *its* parent instead.  Use it when the span's lifetime
    overlaps work that isn't part of it — e.g. a span bracketing an
    async generator handed to a consumer (dispatching tools while a
    model stream is open isn't "inside" the model call), or two
    overlapping spans in the same task.
    """
    if isinstance(name_or_data, str):
        name = name_or_data
        data: SpanData = CustomSpanData(attributes=dict(attributes))
    else:
        if attributes:
            raise TypeError("attributes only go with a str span name")
        name = name_or_data.span_name
        data = name_or_data
    parent = _current.get()
    sp = Span(
        name=name,
        data=data,
        id=messages_.generate_id("span"),
        trace_id=parent.trace_id if parent else messages_.generate_id("trace"),
        parent_id=parent.id if parent else None,
        started_at=time.time_ns(),
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
        sp.ended_at = time.time_ns()
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


# ── wrap_span: one generator frame per span ───────────────────────


class WrapSpanAdapter:
    """Adapter built by :func:`wrap_span`.

    Holds one suspended generator per live span: the frame's locals
    carry state from span start to span end, so context-manager-shaped
    vendor SDKs bridge without any bookkeeping dict.
    """

    def __init__(self, fn: Callable[[Span], AsyncGenerator[None]]) -> None:
        self._fn = fn
        self._live: dict[str, AsyncGenerator[None]] = {}

    def __repr__(self) -> str:
        return f"wrap_span({getattr(self._fn, '__qualname__', self._fn)!r})"

    async def on_span_start(self, span_: Span) -> None:
        gen = self._fn(span_)
        try:
            await anext(gen)
        except StopAsyncIteration:
            return  # returned before yielding: opted out of this span
        self._live[span_.id] = gen

    async def on_span_end(self, span_: Span) -> None:
        gen = self._live.pop(span_.id, None)
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
        raise RuntimeError("wrap_span generator yielded more than once")


def wrap_span(fn: Callable[[Span], AsyncGenerator[None]]) -> WrapSpanAdapter:
    """Build an adapter from an async generator function that yields once.

    The bridge for context-manager-shaped vendor SDKs: write the
    vendor's ``with``/``async with`` around a single ``yield``.  Code
    before the yield runs at span start; code after it runs at span
    end, when ``span.data`` is fully populated and ``ended_at`` is
    set::

        @wrap_span
        async def vendor(span):
            with sdk.start_span(span.name) as v:
                yield
                v.update(output=span.data)

        ai.telemetry.register(vendor)

    - A span that ends with an error is thrown into the generator at
      the ``yield``, so the vendor context manager sees the failure.
      Use ``try/finally`` around the yield for code that must run on
      both paths; catching the error only suppresses it here, never
      for the application.
    - Returning before the first ``yield`` skips that span — cheap
      filtering by span type.
    - Like any adapter, a generator that raises is logged and skipped;
      it never kills the run.
    """
    if not inspect.isasyncgenfunction(fn):
        raise TypeError(
            "wrap_span requires an async generator function "
            "(`async def` containing exactly one bare `yield`)"
        )
    return WrapSpanAdapter(fn)
