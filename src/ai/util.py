"""Utility functions."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import inspect
from typing import TYPE_CHECKING, Any, cast, overload

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Awaitable,
        Callable,
        Collection,
        Generator,
        Iterator,
    )
    from types import TracebackType
    from typing import Protocol

    class ContextManagerWithAsyncDecorator[T](Protocol):
        def __enter__(self) -> T: ...

        def __exit__(
            self,
            typ: type[BaseException] | None,
            value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None: ...

        @overload
        def __call__[**P, R](
            self, func: Callable[P, Awaitable[R]]
        ) -> Callable[P, Awaitable[R]]: ...

        @overload
        def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]: ...


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
        self._tasks: dict[asyncio.Future[T], None] = {}

        # We bind this to an attribute so that the bound method is
        # always the same and can be passed to remove_done_callback.
        self._callback = self._queue.put_nowait
        self.add(*tasks)

    def add(self, *tasks: asyncio.Future[T]) -> None:
        for task in tasks:
            self._tasks[task] = None
            task.add_done_callback(self._callback)

    def clear(self) -> None:
        for task in self._tasks:
            task.remove_done_callback(self._callback)
        self._tasks.clear()

    def tasks(self) -> Collection[asyncio.Future[T]]:
        return self._tasks.keys()

    async def wait(self) -> asyncio.Future[T]:
        t = await self._queue.get()
        self._tasks.pop(t, None)
        return t

    def __await__(self) -> Generator[Any, Any, asyncio.Future[T]]:
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


class _GeneratorContextManagerWithAsyncDecorator[T](
    contextlib._GeneratorContextManager[T]
):
    def __call__[_F: Callable[..., Any]](self, func: _F) -> _F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def inner(*args: Any, **kwds: Any) -> Any:
                with self._recreate_cm():
                    return await func(*args, **kwds)
        else:

            @functools.wraps(func)
            def inner(*args: Any, **kwds: Any) -> Any:
                with self._recreate_cm():
                    return func(*args, **kwds)

        return cast("_F", inner)


def contextmanager_with_async_decorator[**P, T](
    func: Callable[P, Iterator[T]],
) -> Callable[P, ContextManagerWithAsyncDecorator[T]]:
    """@contextmanager decorator but the result can be a decorator for async."""

    @functools.wraps(func)
    def helper(
        *args: P.args, **kwds: P.kwargs
    ) -> _GeneratorContextManagerWithAsyncDecorator[T]:
        return _GeneratorContextManagerWithAsyncDecorator(
            cast("Callable[..., Generator[T, None, None]]", func), args, kwds
        )

    return helper


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
    queue: AsyncIterableQueue[T] = AsyncIterableQueue(size)

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
        async for el in queue:
            yield el
    finally:
        # cancel is a no-op if a task is already done or cancelled
        task.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task


async def merge[T](
    *aiterables: AsyncIterable[T], restart: bool = True
) -> AsyncIterator[T]:
    """Yield elements from async iterables as they arrive.

    Additionally, if `restart` is True (the default), attempt to *restart*
    finished iterables when other iterables produce elements.

    This allows supporting interacting streams, where the processing
    loop might trigger work in one stream based on results from
    another.

    Restarts are only attempted for iterables that are not their own
    iterators (importantly, this means that async generators are not
    restarted).
    """
    async with (
        TaskGroup() as tg,
        MultiWaiter[T]() as mw,
    ):
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
        mw.add(*[t for t in tasks if t])

        top_fired = False
        while mw.tasks():
            t = await mw

            idx = tasks.index(t)
            val = t.result()
            if val is _EMPTY:
                tasks[idx] = None
            else:
                # Fire off a new task for the relevant iterator
                top_fired = True
                iter = aiters[idx]
                tasks[idx] = nt = tg.create_task(anext(iter, _EMPTY))
                mw.add(nt)
                yield val

            if restart and (
                val is not _EMPTY or (not mw.tasks() and top_fired)
            ):
                if not mw.tasks():
                    top_fired = False
                # Also, we try *restarting* other stopped streams
                # that may have more to do now.
                #
                # N.B: We do this *after* the values are yielded, so
                # they've had a chance to trigger things, and we also
                # do it if we would otherwise terminate and we have
                # seen any elements since the start or the last time
                # we may have been exhausted.
                for idx, (ok, otask) in enumerate(
                    zip(restartable, tasks, strict=True)
                ):
                    if ok and otask is None:
                        niter = decouple(aiterables[idx], task_group=tg)
                        aiters[idx] = niter
                        tasks[idx] = nt = tg.create_task(anext(niter, _EMPTY))
                        mw.add(nt)
