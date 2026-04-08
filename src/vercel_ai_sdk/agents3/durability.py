"""Durability provider protocol and built-in EventLog implementation.

A DurabilityProvider intercepts execution boundaries (LLM streams and
tool calls) to record results on fresh execution or replay cached results
on re-entry.  The provider is set on a context var by ``Agent.run()`` and
auto-detected by ``models.stream()`` and ``ToolCall.__call__()``.

Two ways to get durability:

1. **Direct composability** — write a custom loop that wraps primitives in
   your own SDK (Temporal activities, Restate handlers, etc.).  No provider
   needed.

2. **Provider interface** — pass a ``DurabilityProvider`` to ``Agent.run()``.
   The framework routes ``models.stream()`` and ``ToolCall.__call__()``
   through the provider automatically via context var.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Protocol

from .. import _durability as _dctx
from ..types import messages as messages_
from . import checkpoint as checkpoint_

logger = logging.getLogger(__name__)

# Re-export the shared accessors for convenience.
get_provider = _dctx.get_provider
set_provider = _dctx.set_provider
reset_provider = _dctx.reset_provider


# ── Protocol ─────────────────────────────────────────────────────


class StreamResultLike(Protocol):
    """Minimal interface that models.StreamResult satisfies."""

    def __aiter__(self) -> AsyncGenerator[messages_.Message]: ...

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]: ...

    @property
    def text(self) -> str: ...

    @property
    def usage(self) -> messages_.Usage | None: ...


class DurabilityProvider(Protocol):
    """Abstract interface for durable execution of agent operations.

    Implementations intercept LLM streams and tool calls to either
    record results (fresh execution) or replay cached results (re-entry).
    """

    async def execute_stream(
        self,
        fn: Callable[[], Awaitable[StreamResultLike]],
    ) -> StreamResultLike:
        """Wrap an LLM stream step.

        *fn* is an async factory that creates the real ``StreamResult``.
        The provider may call it (fresh execution) or skip it and return
        a ``StreamResult`` wrapping cached messages (replay).

        The provider manages its own step counter internally.
        """
        ...

    async def execute_tool(
        self,
        fn: Callable[[], Awaitable[messages_.ToolResultPart]],
        *,
        tool_call_id: str,
        tool_name: str,
    ) -> messages_.ToolResultPart:
        """Wrap a tool call.

        *fn* executes the real tool.  The provider may call it or return
        a cached ``ToolResultPart`` from the checkpoint.
        """
        ...

    def checkpoint(self) -> checkpoint_.Checkpoint:
        """Return a snapshot of all completed work."""
        ...


# ── EventLogProvider ─────────────────────────────────────────────


class _ReplayStreamResult:
    """Lightweight StreamResult substitute for replayed steps.

    Yields the cached final message and exposes ``.tool_calls``,
    ``.text``, and ``.usage`` just like ``models.StreamResult``.
    """

    def __init__(self, message: messages_.Message) -> None:
        self._message = message

    def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        return self._generate()

    async def _generate(self) -> AsyncGenerator[messages_.Message]:
        yield self._message

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]:
        return self._message.tool_calls

    @property
    def text(self) -> str:
        return self._message.text

    @property
    def usage(self) -> messages_.Usage | None:
        return self._message.usage

    @property
    def output(self) -> Any:
        return self._message.output


class _RecordingStreamResult:
    """Wraps a real StreamResult, forwarding and recording."""

    def __init__(
        self,
        inner: StreamResultLike,
        *,
        index: int,
        steps: list[checkpoint_.StepEvent],
    ) -> None:
        self._inner = inner
        self._index = index
        self._steps = steps
        self._final: messages_.Message | None = None

    def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        return self._generate()

    async def _generate(self) -> AsyncGenerator[messages_.Message]:
        async for msg in self._inner:
            self._final = msg
            yield msg

        # Record the final done message.
        if self._final is not None and self._final.is_done:
            self._steps.append(
                checkpoint_.StepEvent(index=self._index, message=self._final)
            )

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]:
        return self._final.tool_calls if self._final else []

    @property
    def text(self) -> str:
        return self._final.text if self._final else ""

    @property
    def usage(self) -> messages_.Usage | None:
        return self._final.usage if self._final else None

    @property
    def output(self) -> Any:
        return self._final.output if self._final else None


class EventLogProvider:
    """Built-in durability via event log replay.

    Records LLM stream results and tool call results during fresh
    execution.  On re-entry with a checkpoint, replays cached results
    instead of re-executing.

    Usage::

        provider = EventLogProvider(checkpoint)
        async for msg in agent.run(model, messages, durability=provider):
            ...
        new_checkpoint = provider.checkpoint()
    """

    def __init__(self, cp: checkpoint_.Checkpoint | None = None) -> None:
        self._checkpoint = cp or checkpoint_.Checkpoint()
        self._step_cursor: int = 0
        self._tool_cache: dict[str, checkpoint_.ToolEvent] = {
            t.tool_call_id: t for t in self._checkpoint.tools
        }

        # New recordings
        self._steps: list[checkpoint_.StepEvent] = []
        self._tools: list[checkpoint_.ToolEvent] = []

    # ── Stream ────────────────────────────────────────────────

    async def execute_stream(
        self,
        fn: Callable[[], Awaitable[StreamResultLike]],
    ) -> StreamResultLike:
        idx = self._step_cursor
        self._step_cursor += 1

        # Replay from checkpoint.
        if idx < len(self._checkpoint.steps):
            cached = self._checkpoint.steps[idx]
            logger.info("Replaying stream step %d from checkpoint", idx)
            return _ReplayStreamResult(cached.message)

        # Fresh execution — wrap to record.
        return _RecordingStreamResult(await fn(), index=idx, steps=self._steps)

    # ── Tool ──────────────────────────────────────────────────

    async def execute_tool(
        self,
        fn: Callable[[], Awaitable[messages_.ToolResultPart]],
        *,
        tool_call_id: str,
        tool_name: str,
    ) -> messages_.ToolResultPart:
        # Replay from checkpoint.
        cached = self._tool_cache.get(tool_call_id)
        if cached is not None:
            logger.info(
                "Replaying tool %s (call_id=%s) from checkpoint",
                tool_name,
                tool_call_id,
            )
            return messages_.ToolResultPart(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=cached.result,
                is_error=cached.status == "error",
            )

        # Fresh execution.
        result = await fn()
        self._tools.append(
            checkpoint_.ToolEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result.result,
                status="error" if result.is_error else "result",
            )
        )
        return result

    # ── Checkpoint ────────────────────────────────────────────

    def checkpoint(self) -> checkpoint_.Checkpoint:
        """Build a full Checkpoint merging prior state + new recordings."""
        return checkpoint_.Checkpoint(
            steps=list(self._checkpoint.steps) + self._steps,
            tools=list(self._checkpoint.tools) + self._tools,
        )
