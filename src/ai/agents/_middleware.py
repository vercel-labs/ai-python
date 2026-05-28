"""Middleware: composable wrappers around all execution surfaces.

Middleware is run-scoped — pass it to :meth:`Agent.run`::

    agent.run(model, messages, middleware=[LoggingMiddleware()])

Middleware wraps agent runs, model calls, generate calls, tool calls, and
hook calls.  Subclass :class:`Middleware` and override the methods you care
about — unimplemented methods pass through to the next middleware (or the
real implementation).

Ordering: first in the list = outermost.  ``[A(), B()]`` means A wraps B
wraps the real call.  A sees the call first and the result last.
"""

from __future__ import annotations

import contextvars
import dataclasses
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from ..types import messages as messages_

# Compat shim: ``StreamResultLike`` was removed from ``ai.types.proto`` when
# the model layer was reworked.  Middleware is dead code under the new
# ``Executor``-based ``api.py`` and is kept around only so the agents
# rewrite can land separately; ``Any`` is enough to keep the existing
# annotations type-checking.
type StreamResultLike = Any

# ---------------------------------------------------------------------------
# Call context objects — frozen dataclasses with isolated mutable fields.
#
# Mutable container fields (``list``, ``dict``) are shallow-copied at
# construction via ``__post_init__`` so that middleware sees its own copy
# and cannot accidentally mutate the caller's data.  To modify fields,
# use ``dataclasses.replace(call, messages=new_msgs)`` before passing
# to ``next``.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    import pydantic

    from ..models.core.model import Model
    from ..models.core.params import GenerateParams
    from ..types import events as events_
    from ..types.tools import Tool
    from .agent import Context


@dataclasses.dataclass(frozen=True)
class ModelContext:
    """Context for a model streaming call."""

    model: Model
    messages: list[messages_.Message]
    tools: Sequence[Tool] | None
    output_type: type[pydantic.BaseModel] | None
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", list(self.messages))
        if self.tools is not None:
            object.__setattr__(self, "tools", list(self.tools))
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclasses.dataclass(frozen=True)
class GenerateContext:
    """Context for a model generate call (images, video, etc.)."""

    model: Model
    messages: list[messages_.Message]
    params: GenerateParams

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", list(self.messages))


@dataclasses.dataclass(frozen=True)
class ToolContext:
    """Context for a tool execution."""

    tool_call_id: str
    tool_name: str
    kwargs: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclasses.dataclass(frozen=True)
class HookContext:
    """Context for a hook suspension point."""

    label: str
    payload: type[pydantic.BaseModel]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


# ---------------------------------------------------------------------------
# Middleware base class — override the methods you care about.
# ---------------------------------------------------------------------------

# Event/message aliases for brevity in signatures.  ``_Event`` is intentionally
# typed as ``Any`` so the agent-run chain accepts the wider ``AgentEvent``
# union (which includes ``ToolCallResult``/``HookEvent``) without a circular
# import from ``ai.agents``.
_Event = Any
_Message = messages_.Message

# Agent run next-function type: call -> async generator of events.
_AgentRunNext = Callable[["Context"], AsyncGenerator[_Event]]


class _Middleware:
    """Base middleware class.  Override the methods you need.

    Default implementations call ``next(call)`` — a transparent pass-through.
    """

    async def wrap_agent_run(
        self,
        call: Context,
        next: _AgentRunNext,
    ) -> AsyncGenerator[_Event]:
        """Wrap an agent run.

        ``next(call)`` returns an async generator of ``Event`` objects.
        Override to add tracing, durability checkpoints, or other
        run-scoped behavior::

            async def wrap_agent_run(self, call, next):
                span = start_span("agent.run")
                async for event in next(call):
                    yield event
                span.end()
        """
        async for event in next(call):
            yield event

    async def wrap_model(
        self,
        call: ModelContext,
        next: Callable[[ModelContext], Awaitable[StreamResultLike]],
    ) -> StreamResultLike:
        """Wrap a model streaming call.

        ``next(call)`` returns a :class:`~ai.types.StreamResultLike` that
        is async-iterable over ``Event`` objects.  You can do work
        before, iterate / transform the stream, or do cleanup after.

        To transform the stream, use
        :meth:`~ai.models.StreamResult.from_generator`::

            async def wrap_model(self, call, next):
                stream = await next(call)
                async def _add_suffix():
                    async for event in stream:
                        yield event
                from ai.models import StreamResult
                return StreamResult.from_generator(_add_suffix())
        """
        return await next(call)

    async def wrap_generate(
        self,
        call: GenerateContext,
        next: Callable[[GenerateContext], Awaitable[_Message]],
    ) -> _Message:
        """Wrap a model generate call (images, video, etc.)."""
        return await next(call)

    async def wrap_tool(
        self,
        call: ToolContext,
        next: Callable[[ToolContext], Awaitable[events_.ToolCallResult]],
    ) -> events_.ToolCallResult:
        """Wrap a tool execution.

        ``next(call)`` returns a :class:`ToolCallResult`.
        """
        return await next(call)

    async def wrap_hook(
        self,
        call: HookContext,
        next: Callable[[HookContext], Awaitable[pydantic.BaseModel]],
    ) -> pydantic.BaseModel:
        """Wrap a hook suspension point.

        ``next(call)`` blocks until the hook is resolved and returns the
        validated payload instance.
        """
        return await next(call)


# ---------------------------------------------------------------------------
# Run-scoped middleware via ContextVar
# ---------------------------------------------------------------------------

_active: contextvars.ContextVar[list[_Middleware]] = contextvars.ContextVar(
    "middleware",
)

_EMPTY: list[_Middleware] = []


def get() -> list[_Middleware]:
    """Return the middleware stack for the current run (empty if none)."""
    return _active.get(_EMPTY)


Token = contextvars.Token[list[_Middleware]]


def activate(mw: list[_Middleware]) -> Token:
    """Set the middleware stack and return a reset token."""
    return _active.set(mw)


def deactivate(token: Token) -> None:
    """Restore the previous middleware stack."""
    _active.reset(token)


# ---------------------------------------------------------------------------
# Chain builders — compose the middleware stack for each surface.
#
# Each builder takes the *real* implementation as a callable and returns
# a callable with the same signature that routes through middleware.
#
# When no middleware is active, the real implementation is returned
# directly — zero overhead.
# ---------------------------------------------------------------------------


def _build_model_chain(
    real: Callable[[ModelContext], Awaitable[StreamResultLike]],
) -> Callable[[ModelContext], Awaitable[StreamResultLike]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: _Middleware,
            nxt: Callable[[ModelContext], Awaitable[StreamResultLike]],
        ) -> Callable[[ModelContext], Awaitable[StreamResultLike]]:
            async def _wrapped(call: ModelContext) -> StreamResultLike:
                return await m.wrap_model(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_generate_chain(
    real: Callable[[GenerateContext], Awaitable[_Message]],
) -> Callable[[GenerateContext], Awaitable[_Message]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: _Middleware,
            nxt: Callable[[GenerateContext], Awaitable[_Message]],
        ) -> Callable[[GenerateContext], Awaitable[_Message]]:
            async def _wrapped(call: GenerateContext) -> _Message:
                return await m.wrap_generate(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_tool_chain(
    real: Callable[[ToolContext], Awaitable[events_.ToolCallResult]],
) -> Callable[[ToolContext], Awaitable[events_.ToolCallResult]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: _Middleware,
            nxt: Callable[[ToolContext], Awaitable[events_.ToolCallResult]],
        ) -> Callable[[ToolContext], Awaitable[events_.ToolCallResult]]:
            async def _wrapped(call: ToolContext) -> events_.ToolCallResult:
                return await m.wrap_tool(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_hook_chain(
    real: Callable[[HookContext], Awaitable[pydantic.BaseModel]],
) -> Callable[[HookContext], Awaitable[pydantic.BaseModel]]:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(
            m: _Middleware,
            nxt: Callable[[HookContext], Awaitable[pydantic.BaseModel]],
        ) -> Callable[[HookContext], Awaitable[pydantic.BaseModel]]:
            async def _wrapped(call: HookContext) -> pydantic.BaseModel:
                return await m.wrap_hook(call, nxt)

            return _wrapped

        chain = _make(m, chain)
    return chain


def _build_agent_run_chain(
    real: _AgentRunNext,
) -> _AgentRunNext:
    mw = get()
    if not mw:
        return real

    chain = real
    for m in reversed(mw):

        def _make(m: _Middleware, nxt: _AgentRunNext) -> _AgentRunNext:
            async def _wrapped(call: Context) -> AsyncGenerator[_Event]:
                async for event in m.wrap_agent_run(call, nxt):
                    yield event

            return _wrapped

        chain = _make(m, chain)
    return chain


__all__ = [
    "GenerateContext",
    "HookContext",
    "ModelContext",
    "StreamResultLike",
    "ToolContext",
    "_Middleware",
    "activate",
    "deactivate",
    "get",
]
