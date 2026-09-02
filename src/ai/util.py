"""Utility functions."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterable,
        AsyncIterator,
        Collection,
        Generator,
    )
    from types import TracebackType


@dataclasses.dataclass
class _Empty:
    pass


_EMPTY: Any = _Empty()


@dataclasses.dataclass
class _Stop:
    exception: Exception | None = None


_STOP = _Stop()


class AsyncIterableQueue[T](asyncio.Queue[_Stop | T]):
    """An asyncio.Queue that you can iterate over.

    Call athrow or astop to stop it.
    Can not be iterated on by multiple tasks!
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            el = await self.get()
            if isinstance(el, _Stop):
                if el.exception:
                    raise el.exception
                else:
                    return
            yield el

    async def athrow(self, e: Exception) -> None:
        await self.put(_Stop(exception=e))

    async def astop(self) -> None:
        await self.put(_STOP)


class MultiWaiter[T]:
    """Waiter object for waiting on multiple futures.

    The advantages over using asyncio.wait are:
      * New futures may be added while the object is already being waited on
      * Completion order of the tasks is preserved.

    A *potential* downside is:
      * Batching of future completion is lost

    But that is actually good for our use cases, since that introduces
    a potential mismatch when using workflows/temporal.
    """

    def __init__(self, *tasks: asyncio.Future[T]) -> None:
        self._queue: asyncio.Queue[asyncio.Future[T]] = asyncio.Queue(0)
        self._tasks: dict[asyncio.Future[T], Literal[True]] = {}

        # We bind this to an attribute so that the bound method is
        # always the same and can be passed to remove_done_callback.
        self._callback = self._queue.put_nowait
        self.add(*tasks)

    def add(self, *tasks: asyncio.Future[T]) -> None:
        for task in tasks:
            self._tasks[task] = True
            task.add_done_callback(self._callback)

    def discard(self, *tasks: asyncio.Future[T]) -> None:
        for task in tasks:
            self._tasks.pop(task, None)
            task.remove_done_callback(self._callback)
            # Queue it up so that a waiter pops out of the loop
            self._queue.put_nowait(task)

    def clear(self) -> None:
        for task in self._tasks:
            task.remove_done_callback(self._callback)
        self._tasks.clear()

    def tasks(self) -> Collection[asyncio.Future[T]]:
        return self._tasks.keys()

    async def wait(self) -> asyncio.Future[T] | None:
        while self._tasks:
            t = await self._queue.get()
            # Only return the future if it hasn't been discarded
            if self._tasks.pop(t, None):
                return t
        return None

    def __await__(self) -> Generator[Any, Any, asyncio.Future[T] | None]:
        return self.wait().__await__()

    async def __aenter__(self) -> MultiWaiter[T]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> bool:
        self.clear()
        return False


class TaskGroupGenExit(GeneratorExit, BaseExceptionGroup[BaseException]):
    """A ``BaseExceptionGroup`` that is *also* a ``GeneratorExit``.

    Async generator ``aclose()`` only accepts a ``GeneratorExit`` (or
    subclass) propagating out of the generator; a plain
    ``BaseExceptionGroup`` makes it complain and leaves the exception
    unretrieved. By being both, this lets the group satisfy the close
    protocol while still being catchable as the group it really is.
    """


class TaskGroup(asyncio.TaskGroup):
    """asyncio.TaskGroup that directly propagates GeneratorExit.

    If the context body raises a GeneratorExit, we don't want to leave
    it wrapped in a plain ExceptionGroup, because that does the wrong
    thing when it bubbles out through an async generator's aclose().

    So if a GeneratorExit is raised inside the context and that is the
    *only* exception reported, re-raise the group as a TaskGroupGenExit,
    which is *also* a GeneratorExit so aclose() is happy.

    If there are multiple exceptions, keep them packaged in the plain
    group so as to not lose anything (a TaskGroupGenExit would be
    swallowed by aclose(), silently dropping the other exceptions).
    """

    async def __aexit__(
        self,
        et: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            await super().__aexit__(et, exc, tb)
        except BaseExceptionGroup as eg:
            if (
                isinstance(exc, GeneratorExit)
                and len(eg.exceptions) == 1
                and eg.exceptions[0] is exc
            ):
                raise TaskGroupGenExit(
                    eg.message, list(eg.exceptions)
                ) from None
            raise


@contextlib.asynccontextmanager
async def maybe_aclosing(
    iter: AsyncIterable[Any],
) -> AsyncIterator[AsyncIterable[Any]]:
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
    buffer: int | None = 0,
) -> AsyncGenerator[T]:
    """Drive ``iter`` from a single worker task and yield its items.

    Ensures every ``__anext__`` on ``iter`` runs in the same task context, so
    contextvars set or relied on by the iterable behave consistently across
    yields. Without this, callers that wrap each ``anext`` in a fresh task
    (e.g. ``merge``) would run each step in a different copy of the context.

    ``buffer`` is how many elements the worker may run ahead of the
    consumer. With 0, the default, the underlying iterable is run in
    lockstep.

    We try pretty hard to make sure that ``iter`` gets aclose()d in
    the same task that it was run it.

    On asyncio shutdown, tasks all get canceled before async
    generators are closed, so we should be OK.

    """
    queue: AsyncIterableQueue[T] = AsyncIterableQueue()
    sem = None if buffer is None else asyncio.Semaphore(buffer)

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

                # We don't need to wait before the *first* iteration
                # because we don't get spawned until the first anext()
                # anyway.
                async for x in iter:
                    await queue.put(x)
                    if sem is not None:
                        await sem.acquire()
            except Exception as e:
                await queue.put(_Stop(exception=e))
                return
        await queue.put(_STOP)

    if task_group:
        task = task_group.create_task(worker())
    else:
        task = asyncio.create_task(worker())

    try:
        async for el in queue:
            yield el
            if sem is not None:
                sem.release()
    finally:
        # cancel is a no-op if a task is already done or cancelled
        task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task


async def merge[T](
    *aiterables: AsyncIterable[T],
    restart: bool = True,
    priority: bool = False,
) -> AsyncGenerator[T]:
    """Yield elements from async iterables as they arrive.

    The first anext() call on each iterable is done eagerly, but
    after that they run in lockstep with the consumer of merge.

    If `priority` is True (default is False), then earlier async
    iterables take priority over later ones. We will always yield
    a value if available from an earlier one before yielding from
    a later.

    Additionally, if `restart` is True (the default), attempt to *restart*
    finished iterables when other iterables produce elements.

    This allows supporting interacting streams, where the processing
    loop might trigger work in one stream based on results from
    another.

    Restarts are only attempted for iterables that are not their own
    iterators (importantly, this means that async generators are not
    restarted).

    Restart and priority are incompatible.
    """
    if priority and restart:
        raise ValueError("cannot specify priority=True and restart=True")

    async with TaskGroup() as tg:
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
            pending = [t for t in tasks if t]
            done_set, _ = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # We might see an exception before the callback that
            # cancels the main task makes it in, so check.
            if any(t.exception() for t in done_set):
                return

            done = sorted(done_set, key=pending.index)

            fired = []
            for t in done:
                idx = tasks.index(t)
                val = t.result()
                if val is _EMPTY:
                    tasks[idx] = None
                else:
                    yield val
                    # Fire off a new task for the relevant iterator
                    fired.append(idx)
                    iter = aiters[idx]
                    tasks[idx] = tg.create_task(anext(iter, _EMPTY))
                    # sleep(0) to approximate 3.14's eager_start. Make
                    # sure that a trivially read task (like a get() on
                    # a queue with elements) can run to completion.
                    await asyncio.sleep(0)

                if priority:
                    break

            if restart and fired:
                # Also, we try *restarting* other stopped streams
                # that may have more to do now.
                # N.B: We do this *after* the values are yielded, so
                # they've had a chance to trigger things, and we do it
                # after *all* tasks have been handled, so that if a
                # task *just* finished, we still restart it.
                for idx, (ok, otask) in enumerate(
                    zip(restartable, tasks, strict=True)
                ):
                    if ok and otask is None and idx not in fired:
                        niter = aiters[idx] = decouple(
                            aiterables[idx], task_group=tg
                        )
                        tasks[idx] = tg.create_task(anext(niter, _EMPTY))
