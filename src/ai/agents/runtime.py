"""Runtime: message sink that connects producer coroutines to the consumer."""

from __future__ import annotations

import asyncio
import contextvars
from typing import TYPE_CHECKING, Any

from .. import util
from ..types import events as events_
from ..types import messages as messages_
from . import hooks as hooks_
from .mcp import client as mcp_client

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Awaitable


class Runtime:
    """Central event queue. Producers put events, run() yields them."""

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._event_queue: util.AsyncIterableQueue[events_.AgentEvent] = (
            util.AsyncIterableQueue()
        )
        self._hook_labels: set[str] = set()

    async def put_event(self, event: events_.AgentEvent) -> None:
        await self._event_queue.put(event)

    async def put_hook(self, hook_part: messages_.HookPart[Any]) -> None:
        msg = messages_.Message(role="internal", parts=[hook_part])
        await self.put_event(events_.HookEvent(message=msg, hook=hook_part))

    async def signal_done(self) -> None:
        await self._event_queue.astop()

    def track_hook_label(self, label: str) -> None:
        """Register a hook label for cleanup when the run ends."""
        self._hook_labels.add(label)

    def cleanup_hooks(self) -> None:
        """Remove all hook registry entries for this run."""
        hooks_.cleanup_run(self._hook_labels)
        self._hook_labels.clear()


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


def get_runtime() -> Runtime:
    """Return the active Runtime. Raises LookupError outside of run()."""
    return _runtime.get()


async def _stop_when_done(runtime: Runtime, task: Awaitable[None]) -> None:
    try:
        await task
    finally:
        await runtime.signal_done()


async def run(
    source: AsyncIterable[events_.AgentEvent],
) -> AsyncGenerator[events_.AgentEvent]:
    """Run *source* and yield events put into the Runtime queue."""
    rt = Runtime()

    async def _drain() -> None:
        # We do all of the contextvar stuff in _drain so that we don't
        # yield while we have outstanding contextvar manipulations.
        token = _runtime.set(rt)

        # MCP connection pool — scoped to this run.
        mcp_pool: dict[str, mcp_client._Connection] = {}
        mcp_token = mcp_client._pool.set(mcp_pool)

        try:
            async for event in source:
                await rt.put_event(event)

        finally:
            rt.cleanup_hooks()

            await mcp_client.close_connections()
            mcp_client._pool.reset(mcp_token)

            _runtime.reset(token)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_stop_when_done(rt, _drain()))

        async for item in rt._event_queue:
            yield item
