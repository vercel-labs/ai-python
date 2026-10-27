"""Utility functions."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import functools
import weakref
from collections.abc import AsyncGenerator, Callable, MutableSet
from typing import TYPE_CHECKING, Any, Protocol

import anyio
import sniffio

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Collection,
        Coroutine,
        Generator,
        Iterable,
        Iterator,
    )
    from types import TracebackType


@dataclasses.dataclass
class _Empty:
    pass


_EMPTY: Any = _Empty()


@dataclasses.dataclass
class _Stop:
    exception: BaseException | None = None


_STOP = _Stop()


class AsyncIterableQueue[T](asyncio.Queue[_Stop | T]):
    """An asyncio.Queue that you can iterate over.

    Call athrow or astop to stop it.
    Can not be iterated on by multiple tasks!
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        el = await self.get()
        if isinstance(el, _Stop):
            if el.exception:
                raise el.exception
            else:
                raise StopAsyncIteration
        return el

    async def athrow(self, e: BaseException) -> None:
        await self.put(_Stop(exception=e))

    async def astop(self) -> None:
        await self.put(_STOP)


class Waitable(Protocol):
    async def wait(self) -> object: ...


class MultiWaiter[T: Waitable]:
    """Waiter object for waiting on multiple waitables.

    Anything with an async ``wait()`` method works: ``anyio.Event``,
    ``anyio.TaskHandle``, etc. Must be entered as an async context
    manager, since each item is waited on by a task of its own.

    The advantages over using asyncio.wait are:
      * New items may be added while the object is already being waited on
      * Completion order of the items is preserved.

    A *potential* downside is:
      * Batching of completion is lost

    But that is actually good for our use cases, since that introduces
    a potential mismatch when using workflows/temporal.
    """

    def __init__(self, *items: T) -> None:
        self._queue: collections.deque[T] = collections.deque()
        self._items: dict[T, anyio.TaskHandle[None] | None] = {}
        self._wakeup: anyio.Event | None = None
        self._tg: anyio.abc.TaskGroup | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        self.add(*items)

    async def _watch(self, item: T) -> None:
        await item.wait()
        self._queue.append(item)
        self._notify()

    def _notify(self) -> None:
        if self._wakeup is not None:
            self._wakeup.set()

    def add(self, *items: T) -> None:
        for item in items:
            self._items[item] = (
                self._tg.create_task(self._watch(item)) if self._tg else None
            )

    def discard(self, *items: T) -> None:
        for item in items:
            if item in self._items:
                if handle := self._items.pop(item):
                    handle.cancel()
                # Wake up a waiter so that it can pop out of the loop
                self._notify()

    def clear(self) -> None:
        for handle in self._items.values():
            if handle:
                handle.cancel()
        self._items.clear()
        self._queue.clear()
        self._notify()

    def tasks(self) -> Collection[T]:
        return self._items.keys()

    async def wait(self) -> T | None:
        while self._items:
            if not self._queue:
                self._wakeup = anyio.Event()
                await self._wakeup.wait()
                continue
            t = self._queue.popleft()
            # Only return the item if it hasn't been discarded
            if t in self._items:
                del self._items[t]
                return t
        return None

    def __await__(self) -> Generator[Any, Any, T | None]:
        return self.wait().__await__()

    async def __aenter__(self) -> MultiWaiter[T]:
        self._tg = await self._exit_stack.enter_async_context(
            create_task_group()
        )
        for item, handle in self._items.items():
            if handle is None:
                self._items[item] = self._tg.create_task(self._watch(item))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> bool:
        self.clear()
        self._tg = None
        # The watchers are all cancelled, so don't hand the exception
        # to the task group; it would just get wrapped in a group.
        await self._exit_stack.aclose()
        return False


class OrderedSet[T](MutableSet[T]):
    """An insertion-ordered set."""

    def __init__(self, iterable: Iterable[T] = ()) -> None:
        self._items: dict[T, None] = dict.fromkeys(iterable)

    def add(self, value: T) -> None:
        self._items[value] = None

    def discard(self, value: T) -> None:
        self._items.pop(value, None)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class TaskGroup(asyncio.TaskGroup):
    """TaskGroup that propagates GeneratorExit and has deterministic teardown.

    If the context body raises a GeneratorExit, we don't want to leave
    it wrapped in a plain ExceptionGroup, because that does the wrong
    thing when it bubbles out through an async generator's aclose().

    So if a GeneratorExit is raised inside the context and that is the
    *only* exception reported, re-raise the GeneratorExit itself.

    If there are multiple exceptions, keep them packaged in the group so
    as to not lose anything (a bare GeneratorExit would be swallowed by
    aclose(), silently dropping the other exceptions).

    On exceptional exit, tasks are cancelled in the order they were
    created.
    """

    def __init__(self) -> None:
        super().__init__()
        # Bang in an ordered set so we tear down in order.
        self._tasks = OrderedSet()  # type: ignore  # noqa: PGH003

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
                raise exc from None
            raise


@contextlib.asynccontextmanager
async def create_task_group() -> AsyncIterator[anyio.abc.TaskGroup]:
    """Make an anyio TaskGroup with GeneratorExit handling and ordered teardown.

    If the context body raises a GeneratorExit, we don't want to leave
    it wrapped in a plain ExceptionGroup, because that does the wrong
    thing when it bubbles out through an async generator's aclose().

    So if a GeneratorExit is raised inside the context and that is the
    *only* exception reported, re-raise the GeneratorExit itself.

    If there are multiple exceptions, keep them packaged in the group so
    as to not lose anything (a bare GeneratorExit would be swallowed by
    aclose(), silently dropping the other exceptions).

    On exceptional exit with an asyncio backend, tasks are cancelled
    in the order they were created. (Trio randomizes the scheduling
    order anyway so god help you.)
    """
    tg = anyio.create_task_group()
    if sniffio.current_async_library() == "asyncio":
        # Bang in ordered sets so we tear down in order.
        atg: Any = tg
        atg._tasks = OrderedSet()
        atg.cancel_scope._tasks = OrderedSet()
        atg.cancel_scope._child_scopes = OrderedSet()

    body_exc: BaseException | None = None
    try:
        async with tg:
            try:
                yield tg
            except BaseException as e:
                body_exc = e
                raise
    except BaseExceptionGroup as eg:
        if (
            isinstance(body_exc, GeneratorExit)
            and len(eg.exceptions) == 1
            and eg.exceptions[0] is body_exc
        ):
            raise body_exc from None
        raise


def _is_loop_running() -> bool:
    try:
        sniffio.current_async_library()
        return True
    except sniffio.AsyncLibraryNotFoundError:
        return False


@contextlib.contextmanager
def _filter_generator_exit() -> Iterator[None]:
    def strip(eg: BaseExceptionGroup[Any]) -> BaseExceptionGroup[Any] | None:
        kept: list[BaseException] = []
        for exc in eg.exceptions:
            if isinstance(exc, BaseExceptionGroup):
                if (sub := strip(exc)) is not None:
                    kept.append(sub)
            elif not isinstance(exc, GeneratorExit):
                kept.append(exc)
        return eg.derive(kept) if kept else None

    try:
        yield
    except BaseExceptionGroup as eg:
        if (rest := strip(eg)) is not None:
            raise rest from eg


class _AsyncGenProxy[Y, S](AsyncGenerator[Y, S]):
    def __init__(self, gen: AsyncGenerator[Y, S]) -> None:
        self._gen = gen

    def __aiter__(self) -> AsyncGenerator[Y, S]:
        return self

    def __anext__(self) -> Coroutine[Any, Any, Y]:
        return self._gen.__anext__()

    def asend(self, value: S) -> Coroutine[Any, Any, Y]:
        return self._gen.asend(value)

    def athrow(self, *args: Any) -> Coroutine[Any, Any, Y]:
        return self._gen.athrow(*args)

    async def aclose(self) -> None:
        with _filter_generator_exit():
            await self._gen.aclose()


def filter_generator_exit[**P, Y, S](
    func: Callable[P, AsyncGenerator[Y, S]],
) -> Callable[P, _AsyncGenProxy[Y, S]]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> _AsyncGenProxy[Y, S]:
        return _AsyncGenProxy(func(*args, **kwargs))

    return wrapper


_LOOP_CLOSING_MAP: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Task[None]
] = weakref.WeakKeyDictionary()


def get_loop_closing_checker() -> Callable[[], bool]:
    """Get a function that checks if the current loop has started closing.

    It only is guaranteed to detect a close that started *after* the call,
    since we lazily construct the structure used to do it.

    More precisely, the returned function returns whether a dummy task
    has been cancelled, which in practice will only occur if something
    cancels *every* task, which occurs when `run` is torn down.
    """

    async def forever() -> None:
        await asyncio.Future()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return lambda: False
    assert loop
    if not (t := _LOOP_CLOSING_MAP.get(loop)):
        t = _LOOP_CLOSING_MAP[loop] = asyncio.create_task(
            forever(), name="closing-dummy"
        )
        # The task references the loop, so drop the entry ourselves.
        t.add_done_callback(lambda _: _LOOP_CLOSING_MAP.pop(loop, None))
    # Basically the idea here is that the only way this dummy future
    # will ever get cancelled is if *all* futures are cancelled.
    return lambda: t.cancelling() > 0 or t.cancelled()


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


def run_right_now[T](fut: asyncio.Future[T], val: T) -> bool:
    """Signal a future and try to run a task blocked on it *right now*.

    This can give big (~2x) speedups on tight loops ping-ponging
    between two tasks, since it avoids going through the event loop
    scheduler.
    """
    if fut.done():
        return False

    # As a (10%?) microoptimization for the 1-callback common case, we
    # don't implement this for > 1.
    # If anybody cared it could be done in a specialized branch.
    if not fut._callbacks or len(fut._callbacks) != 1:
        fut.set_result(val)
        return False

    callback, context = fut._callbacks[0]
    # Blow away the callback, since we are calling it ourselves.
    fut.remove_done_callback(callback)
    fut.set_result(val)

    loop = asyncio.get_running_loop()
    cur = asyncio.current_task()
    assert cur
    asyncio._leave_task(loop, cur)
    try:
        context.run(callback, fut)
    finally:
        asyncio._enter_task(loop, cur)

    return True


class decouple[T]:  # noqa: N801
    """Drive ``iter`` from a single worker task and yield its items.

    Ensures every ``__anext__`` on ``iter`` runs in the same task context,
    which makes it safe to call ``anext`` on a ``decouple`` from
    different tasks. (Async generators may depend on both context vars
    and the current task identity, so in general should be run on one task.)

    decouple takes ownership of the iterable, and will call aclose() on it if
    aclose() exists.

    anext() on a decouple is cancellation-safe (similar to Queue.get):
    cancelling it will not lose elements.

    ``buffer`` is how many elements the worker may run ahead of the
    consumer. With buffer=0 the underlying iterable is run in
    lockstep with the consumer.

    If iter does *not* have an aclose method, then the underlying
    iterator ``iter`` will be usable after the decouple is closed. If
    the buffer size was zero, then no elements will be lost unless an
    anext was cancelled, in which case one might be.

    """

    def __init__(
        self,
        iter: AsyncIterable[T],
        *,
        buffer: int | None,
        task_group: anyio.abc.TaskGroup | None = None,
    ) -> None:
        # How many anexts have been cancelled - avoid signalling the sem
        # when they are.
        self._cancelled_nexts = 0

        self._iter = iter
        self._buffer = buffer
        self._task_group = task_group
        self._exit_stack = contextlib.AsyncExitStack()

    async def __aenter__(self) -> decouple[T]:
        await self._start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _start(self) -> None:
        if not hasattr(self, "_worker"):
            tg = self._task_group
            if tg is None:
                tg = await self._exit_stack.enter_async_context(
                    anyio.create_task_group()
                )
            queue: collections.deque[_Stop | T] = collections.deque()
            recv_sem = anyio.Semaphore(0, fast_acquire=True)
            self._queue = queue
            self._recv_sem = recv_sem

            def put(x: _Stop | T) -> None:
                recv_sem.release()
                queue.append(x)

            self._worker = _DecoupleWorker(tg, self._iter, self._buffer, put)

    async def __anext__(self) -> T:
        await self._start()
        try:
            if self._cancelled_nexts:
                self._cancelled_nexts -= 1
            else:
                self._worker.release()

            await self._recv_sem.acquire()
            item = self._queue.popleft()

            if isinstance(item, _Stop):
                if item.exception is not None:
                    raise item.exception
                raise StopAsyncIteration
            return item
        except asyncio.CancelledError:
            self._cancelled_nexts += 1
            raise

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def aclose(self) -> None:
        await self._start()
        try:
            await self._worker.aclose()
        finally:
            # genuinely we don't care
            with contextlib.suppress(BaseExceptionGroup):
                await self._exit_stack.aclose()

    def __del__(self) -> None:
        # Dropped without aclose(): stop the worker so it closes iter
        # in its own task, like an async generator's finalizer would.
        worker = getattr(self, "_worker", None)
        if worker is not None and _is_loop_running():
            worker.stop()

    async def asend(self, value: Any, /) -> T:
        raise RuntimeError("decouple does not support asend()")

    async def athrow(self, *args: Any) -> T:
        raise RuntimeError("decouple does not support athrow()")


def is_anyio_cancellation(exc: asyncio.CancelledError) -> bool:
    # Sometimes third party frameworks catch a CancelledError and
    # raise a new one, so as a workaround we have to look at the
    # previous ones in __context__ too for a matching cancel message
    while True:
        if (
            exc.args
            and isinstance(exc.args[0], str)
            and exc.args[0].startswith("Cancelled via cancel scope ")
        ):
            return True

        if isinstance(exc.__context__, asyncio.CancelledError):
            exc = exc.__context__
            continue

        return False


class _DecoupleWorker[T]:
    def __init__(
        self,
        tg: anyio.abc.TaskGroup,
        iter: AsyncIterable[T],
        buffer: int | None,
        put: Callable[[_Stop | T], None],
    ) -> None:
        self._put = put
        self._sem = (
            None
            if buffer is None
            else anyio.Semaphore(buffer, fast_acquire=True)
        )
        self._done = False
        self._is_loop_closing = get_loop_closing_checker()
        self.task = tg.create_task(self._run(iter), name=f"decouple for {iter}")

        self._use_fut = (
            buffer == 0 and sniffio.current_async_library() == "asyncio"
        )
        self._cur_fut: asyncio.Future[None] | None = None

    def release(self) -> bool:
        ran_eagerly = False
        if self._sem is not None:
            self._sem.release()
            if self._cur_fut:
                ran_eagerly = run_right_now(self._cur_fut, None)
        return ran_eagerly

    def stop(self) -> None:
        self._done = True
        if self._sem is not None:
            self._sem.release()
        if self._cur_fut and not self._cur_fut.done():
            self._cur_fut.set_result(None)
        # cancel is a no-op if a task is already done or cancelled
        # XXX: want to resotre this...
        # if not self.task.cancelling():
        self.task.cancel()

    async def aclose(self) -> None:
        self.stop()
        try:
            # XXX: don't really need this
            with contextlib.suppress(anyio.get_cancelled_exc_class()):
                await self.task
            # XXX: wait, I had kind of been assuming no worker
            # failures, but they can fail on the aclose!!
            if self.task.exception:
                raise self.task.exception
        except anyio.TaskCancelled:
            pass
        except anyio.TaskFailed as e:
            assert e.__cause__ is not None
            raise e.__cause__ from None

    async def _acquire(self) -> None:
        if self._sem is None:
            return
        try:
            # For the running in lock-step case, we go to sleep on a
            # future that we can run with run_right_now(), which
            # allows us to run it without hitting the scheduler.
            #
            # If we manage to produce a value without blocking, then
            # by the time we block again (back on this future), we'll
            # have already populated the queue and the consumer will
            # be able to read it without ever blocking either, so we
            # shave two trips through the scheduler.
            if self._use_fut and self._sem.value == 0:
                self._cur_fut = asyncio.Future()
                try:
                    await self._cur_fut
                finally:
                    self._cur_fut = None

            # In general, we need to shield this acquire because we
            # don't want an anyio cancellation that occurs in the loop
            # body to mess us up... but setting up a CancelScope is
            # hella slow, so we just don't, and then if we have to
            # retry things in the exception handler, we scope there.
            await self._sem.acquire()
        except anyio.get_cancelled_exc_class() as e:
            # Three reasons we might have been cancelled:
            # 1. aclose()
            # 2. Approximately *all* tasks being cancelled
            # 3. Something internal to the generator body
            #    (probably a TaskGroup)
            # 4. anyio nested cancellation
            #
            # In case 3, we wait again on the sem (to
            # preserve lockstep behavior), then we
            # re-assert the cancellation so it gets
            # delivered back into the generator body
            # if it blocks.
            #
            # In case 4 we do the same, and it should be fine, at
            # least if everything is nested properly?
            if self._done or self._is_loop_closing():
                raise
            # We do need a CancelScope here, to protect from nested whatevers...
            with (
                anyio.CancelScope(shield=True),
                contextlib.suppress(asyncio.CancelledError),
            ):
                await self._sem.acquire()

            if isinstance(
                e, asyncio.CancelledError
            ) and not is_anyio_cancellation(e):
                # For asyncio cancellations, it is edge triggered, so
                # we need to recancel the task, so that it gets
                # delivered, but then also *uncancel* it, so the count
                # doesn't go up.
                task = asyncio.current_task()
                assert task
                task.cancel(str(e))
                task.uncancel()

    async def _run(self, iter: AsyncIterable[T]) -> None:
        async with maybe_aclosing(iter):
            try:
                await self._acquire()
                async for x in iter:
                    self._put(x)
                    await self._acquire()
                    if self._done:
                        break
            except (Exception, BaseExceptionGroup) as e:
                self._put(_Stop(exception=e))
            except asyncio.CancelledError as e:
                task = asyncio.current_task()
                assert task
                if task.cancelling():
                    # Someone is actually cancelling the worker.
                    self._put(_STOP)
                    raise
                # A cancel came out of the iterator without anyone
                # cancelling the worker. That's probably the < 3.13
                # uncancel() misbehavior, where a cancel we re-armed
                # stays pending even after the count drops back to 0.
                # Don't re-raise a cancel outside of its scope in the
                # consumer; report it as an error instead.
                err = RuntimeError(
                    "iterator raised CancelledError without the decouple "
                    "worker being cancelled"
                )
                err.__cause__ = e
                self._put(_Stop(exception=err))
            else:
                self._put(_STOP)


class AsyncContextManagerGenerator[YieldT, SendT](
    AsyncGenerator[YieldT, SendT]
):
    def __init__(self, iter: AsyncGenerator[YieldT, SendT]) -> None:
        self._iter = iter

    def __aiter__(self) -> AsyncIterator[YieldT]:
        return self

    async def __anext__(self) -> YieldT:
        return await anext(self._iter)

    async def aclose(self) -> None:
        await self._iter.aclose()

    async def asend(self, value: SendT, /) -> YieldT:
        return await self._iter.asend(value)

    async def athrow(self, *args: Any) -> YieldT:
        return await self._iter.athrow(*args)

    async def __aenter__(
        self,
    ) -> AsyncContextManagerGenerator[YieldT, SendT]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def merge[T](
    *aiterables: AsyncIterable[T],
    restart: bool = True,
    priority: bool = False,
) -> AsyncContextManagerGenerator[T, None]:
    return AsyncContextManagerGenerator(
        _merge(*aiterables, restart=restart, priority=priority)
    )


async def _merge[T](
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

    exc: BaseExceptionGroup | None = None

    # Elements that workers have produced, as (index, element), and an
    # event that gets set when one is added.
    ready: list[tuple[int, _Stop | T]] = []
    wakeup = anyio.Event()

    def start(idx: int, iterable: AsyncIterable[T]) -> _DecoupleWorker[T]:
        def put(x: _Stop | T) -> None:
            ready.append((idx, x))
            wakeup.set()

        worker = _DecoupleWorker(worker_tg, iterable, 0, put)
        worker.release()
        return worker

    async with (
        create_task_group() as worker_tg,
        contextlib.AsyncExitStack() as stack,
    ):
        raw_aiters = [aiter(iter) for iter in aiterables]
        workers = [start(idx, iter) for idx, iter in enumerate(raw_aiters)]
        running = [True] * len(workers)

        @stack.push_async_callback
        async def _close_workers() -> None:
            for worker in workers:
                await worker.aclose()

        # We consider anything that doesn't __aiter__ to itself to be
        # potentially restartable.
        restartable = [
            aiterable is not aiterator
            for aiterable, aiterator in zip(aiterables, raw_aiters, strict=True)
        ]

        while any(running):
            if not ready:
                await wakeup.wait()
                wakeup = anyio.Event()

            if errors := [
                val.exception
                for _, val in ready
                if isinstance(val, _Stop) and val.exception is not None
            ]:
                exc = BaseExceptionGroup("unhandled errors in merge", errors)
                break

            ready.sort(key=lambda r: r[0])
            n = 1 if priority else len(ready)
            done = ready[:n]
            del ready[:n]

            fired = []
            for idx, val in done:
                if isinstance(val, _Stop):
                    running[idx] = False
                else:
                    yield val
                    # Ask the relevant iterator for its next element
                    fired.append(idx)
                    ran_eagerly = workers[idx].release()
                    if priority:
                        # Make sure that a trivially ready element (like a
                        # get() on a queue with elements) gets produced
                        # before we look at what's ready again.
                        if not ran_eagerly:
                            await anyio.sleep(0)
                        # ... and once more so the worker goes
                        await anyio.sleep(0)

            if restart and fired:
                # Also, we try *restarting* other stopped streams
                # that may have more to do now.
                # N.B: We do this *after* the values are yielded, so
                # they've had a chance to trigger things, and we do it
                # after *all* ready elements have been handled, so
                # that if an iterator *just* finished, we still
                # restart it.
                for idx, (ok, alive) in enumerate(
                    zip(restartable, running, strict=True)
                ):
                    if ok and not alive and idx not in fired:
                        workers[idx] = start(idx, aiterables[idx])
                        running[idx] = True

    if exc:
        raise exc
