"""Runtime: message sink that connects producer coroutines to the consumer."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable

from .. import types


class Runtime:
    """Central message queue. Producers put messages, run() yields them."""

    class _Sentinel:
        pass

    _SENTINEL = _Sentinel()

    def __init__(self) -> None:
        self._message_queue: asyncio.Queue[types.Message | Runtime._Sentinel] = (
            asyncio.Queue()
        )

    async def put_message(self, message: types.Message) -> None:
        await self._message_queue.put(message)

    async def signal_done(self) -> None:
        await self._message_queue.put(self._SENTINEL)


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
    source: AsyncIterable[types.Message],
) -> AsyncGenerator[types.Message]:
    """Run *source* and yield every message that gets put into the Runtime queue."""

    runtime = Runtime()
    token = _runtime.set(runtime)

    async def _drain() -> None:
        async for message in source:
            await runtime.put_message(message)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_stop_when_done(runtime, _drain()))

            while True:
                item = await runtime._message_queue.get()
                if isinstance(item, Runtime._Sentinel):
                    return
                yield item

    finally:
        _runtime.reset(token)
