"""Runtime: message sink that connects producer coroutines to the consumer."""

from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING, Any

from .. import util
from ..types import events as events_
from ..types import messages as messages_
from .mcp import client as mcp_client

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class Runtime:
    """Central event queue. Producers put events, run() yields them."""

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._event_queue: util.AsyncIterableQueue[events_.AgentEvent] = (
            util.AsyncIterableQueue()
        )

    async def put_event(self, event: events_.AgentEvent) -> None:
        await self._event_queue.put(event)

    async def put_hook(self, hook_part: messages_.HookPart[Any]) -> None:
        msg = messages_.Message(role="internal", parts=[hook_part])
        await self.put_event(events_.HookEvent(message=msg, hook=hook_part))

    async def signal_done(self) -> None:
        await self._event_queue.astop()


_runtime: contextvars.ContextVar[Runtime] = contextvars.ContextVar("runtime")


def get_runtime() -> Runtime:
    """Return the active Runtime. Raises LookupError outside of run()."""
    return _runtime.get()


def run(
    source: AsyncGenerator[events_.AgentEvent],
) -> AsyncGenerator[events_.AgentEvent]:
    """Run *source* and yield events put into the Runtime queue."""
    rt = Runtime()

    async def _drain() -> AsyncGenerator[events_.AgentEvent]:
        # We do all of the contextvar stuff in _drain so that we don't
        # yield while we have outstanding contextvar manipulations.
        token = _runtime.set(rt)

        # MCP connection pool — scoped to this run.
        mcp_pool: dict[str, mcp_client._Connection] = {}
        mcp_token = mcp_client._pool.set(mcp_pool)

        try:
            async with contextlib.aclosing(source) as events:
                async for event in events:
                    yield event

        finally:
            await mcp_client.close_connections()
            mcp_client._pool.reset(mcp_token)

            _runtime.reset(token)

            await rt.signal_done()

    # Merge while prioritizing the _event_queue, so that partial tool
    # results will always precede the real tool result.
    return util.merge(rt._event_queue, _drain(), restart=False, priority=True)
