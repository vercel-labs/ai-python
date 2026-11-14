"""Helpers for :mod:`ai.contrib.sandbox`."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable
from contextlib import suppress
from typing import cast

import anyio

__all__ = ["protected"]


async def protected[T](arg: Awaitable[T]) -> T:
    """Wait for a future, deferring cancellation until it is complete.

    If you do
        await protected(something())

    it is approximately equivalent to
        await something()

    except that if the coroutine containing it is cancelled,
    something() is protected from cancellation, and *additionally*
    CancelledError will not be delivered to the caller yet.
    If the caller gets cancelled while executing this, however,
    the cancellation state *will* remain armed, so that cancellation
    should be delivered the next time it blocks

    If anyio is imported, we will enter an anyio CancelScope. As an
    optimization, we only do it *after* but only *after* we take a
    CancelledError.

    This is useful for shielding cleanup code from cancellation,
    especially code in finally blocks.

    """
    loop = None
    with suppress(RuntimeError):
        loop = asyncio.get_running_loop()

    if not loop:
        with anyio.CancelScope(shield=True):
            return await arg

    done = False
    result: T | None = None
    exc: BaseException | None = None
    waiter: asyncio.Future[None] | None = None

    async def go() -> None:
        nonlocal done, result, exc
        try:
            result = await arg
        except BaseException as e:
            exc = e
            # swallow
        finally:
            done = True
            if waiter and not waiter.done():
                waiter.set_result(None)

    # The event loop only holds tasks weakly.
    _task = asyncio.Task(go(), loop=loop, eager_start=True)

    cancelled: asyncio.CancelledError | None = None
    scope: anyio.CancelScope | None = None
    try:
        while not done:
            try:
                waiter = asyncio.Future()
                await waiter
            except asyncio.CancelledError as cex:
                if not scope:
                    scope = anyio.CancelScope(shield=True)
                    scope.__enter__()

                cancelled = cex
    finally:
        if scope:
            scope.__exit__(None, None, None)

    # anyio redelivers its own cancellations while the scope stays
    # cancelled.  Rearming one anyway would leak it past the scope on 3.12,
    # where uncancel() doesn't clear a pending cancel.
    if cancelled and anyio.current_effective_deadline() != -math.inf:
        current = asyncio.current_task()
        assert current
        # For asyncio cancellations, it is edge triggered, so
        # we need to recancel the task, so that it gets
        # delivered, but then also *uncancel* it, so the count
        # doesn't go up.
        current.cancel(cancelled.args[0] if cancelled.args else None)
        current.uncancel()

    if exc:
        raise exc
    return cast("T", result)
