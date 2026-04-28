"""Utility functions"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator


async def merge[T](*aiterables: AsyncIterable[T]) -> AsyncIterator[T]:
    aiters = [aiter(iter) for iter in aiterables]

    # Launch a task doing anext on every iterator
    tasks: list[asyncio.Future[T] | None] = [
        asyncio.ensure_future(anext(iter)) for iter in aiters
    ]

    try:
        while any(tasks):
            done, _ = await asyncio.wait(
                [t for t in tasks if t],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in done:
                idx = tasks.index(t)
                # Note: .exception() could raise CancelledError
                if exc := t.exception():
                    # Happy case for exception is StopAsyncIteration
                    # For other exceptions, raise
                    tasks[idx] = None
                    if not isinstance(exc, StopAsyncIteration):
                        raise exc
                else:
                    # Fire off a new task for the relevant iterator
                    iter = aiters[idx]
                    tasks[idx] = asyncio.ensure_future(anext(iter))
                    yield t.result()
    except Exception:
        for task in tasks:
            if task:
                task.cancel()

        live = [t for t in tasks if t]
        await asyncio.gather(*live, return_exceptions=True)

        raise
