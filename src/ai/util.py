"""Utility functions"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

_EMPTY: Any = object()


@dataclasses.dataclass
class _Stop:
    exception: Exception | None = None


_STOP = _Stop()


@contextlib.asynccontextmanager
async def unwrap_generator_exit() -> AsyncIterator[None]:
    """Convert a ``BaseExceptionGroup`` of only ``GeneratorExit``s into a single one.

    ``asyncio.TaskGroup``'s ``__aexit__`` wraps any body exception (including
    ``GeneratorExit``) into a ``BaseExceptionGroup``. Inside an async
    generator that means ``aclose()`` propagates an ``ExceptionGroup`` instead
    of the bare ``GeneratorExit`` the protocol expects, and the aclose-task
    ends up with an unretrieved exception. Wrapping the ``async with
    TaskGroup(...)`` block in this manager unwraps the group back to a plain
    ``GeneratorExit`` so the close path stays clean.
    """
    try:
        yield
    except BaseExceptionGroup as eg:
        matched, rest = eg.split(GeneratorExit)
        if matched is not None and rest is None:
            raise GeneratorExit from None
        raise


@contextlib.asynccontextmanager
async def maybe_aclosing[T: AsyncIterable[Any]](iter: T) -> AsyncIterator[T]:
    """Like ``contextlib.aclosing`` but a no-op if ``iter`` has no ``aclose``.

    Useful when consuming an arbitrary ``AsyncIterable[T]`` whose concrete
    type may or may not be an async generator.
    """
    try:
        yield iter
    finally:
        aclose = getattr(iter, "aclose", None)
        if aclose is not None:
            await aclose()


async def decouple[T](
    iter: AsyncIterable[T],
    *,
    task_group: asyncio.TaskGroup | None,
    size: int = 1,
) -> AsyncIterator[T]:
    """Drive ``iter`` from a single worker task and yield its items.

    Ensures every ``__anext__`` on ``iter`` runs in the same task context, so
    contextvars set or relied on by the iterable behave consistently across
    yields. Without this, callers that wrap each ``anext`` in a fresh task
    (e.g. ``merge``) would run each step in a different copy of the context.

    We try pretty hard to make sure that ``iter`` gets aclose()d in
    the same task that it was run it.

    On asyncio shutdown, tasks all get canceled before async
    generators are closed, so we should be OK.

    """
    queue: asyncio.Queue[_Stop | T] = asyncio.Queue(size)

    async def worker() -> None:
        async with maybe_aclosing(iter):
            try:
                # N.B: There's a potential case, if iter is *not* a
                # generator (and so we aren't closing it), and this
                # task gets cancelled before it can write it, then
                # maybe an element gets lost?
                #
                # TODO: I'm not sure if this case can ever matter, but
                # think about it more.
                async for x in iter:
                    await queue.put(x)
            except Exception as e:
                await queue.put(_Stop(exception=e))
                return
        await queue.put(_STOP)

    if task_group:
        task = task_group.create_task(worker())
    else:
        task = asyncio.create_task(worker())

    try:
        while True:
            el = await queue.get()
            if isinstance(el, _Stop):
                if el.exception:
                    raise el.exception
                else:
                    return
            yield el
    finally:
        # cancel is a no-op if a task is already done or cancelled
        task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task


async def merge[T](
    *aiterables: AsyncIterable[T], restart: bool = True
) -> AsyncIterator[T]:
    """Produce a stream that yields elements from async iterables as they arrive.

    Additionally, if `restart` is True (the default), attempt to *restart*
    finished iterables when other iterables produce elements.

    This allows supporting interacting streams, where the processing
    loop might trigger work in one stream based on results from
    another.

    Restarts are only attempted for iterables that are not their own
    iterators (importantly, this means that async generators are not
    restarted).
    """

    # We use unwrap_generator_exit() to keep a GeneratorExit that gets
    # packaged in an ExceptionGroup from causing grief. But maybe we
    # ought to not use a TaskGroup?
    async with unwrap_generator_exit(), asyncio.TaskGroup() as tg:
        raw_aiters = [aiter(iter) for iter in aiterables]
        aiters = [decouple(iter, task_group=tg) for iter in raw_aiters]
        # We consider anything that doesn't __aiter__ to itself to be
        # potentially restartable.
        restartable = [
            aiterable is not aiterator
            for aiterable, aiterator in zip(aiterables, raw_aiters, strict=True)
        ]

        # Launch a task doing anext on every iterator
        tasks: list[asyncio.Future[T] | None] = [
            tg.create_task(anext(iter, _EMPTY)) for iter in aiters
        ]

        while any(tasks):
            done, _ = await asyncio.wait(
                [t for t in tasks if t],
                return_when=asyncio.FIRST_COMPLETED,
            )

            fired = []
            for t in done:
                idx = tasks.index(t)
                val = t.result()
                if val is _EMPTY:
                    tasks[idx] = None
                else:
                    # Fire off a new task for the relevant iterator
                    fired.append(idx)
                    iter = aiters[idx]
                    tasks[idx] = tg.create_task(anext(iter, _EMPTY))
                    yield val

            if restart and fired:
                # Also, we try *restarting* other stopped streams
                # that may have more to do now.
                # N.B: We do this *after* the values are yielded, so
                # they've had a chance to trigger things, and we do it
                # after *all* tasks have been handled, so that if a
                # task *just* finished, we still restart it.
                for idx, (ok, otask) in enumerate(zip(restartable, tasks, strict=True)):
                    if ok and otask is None and idx not in fired:
                        niter = decouple(aiterables[idx], task_group=tg)
                        aiters[idx] = niter
                        tasks[idx] = tg.create_task(anext(niter, _EMPTY))
