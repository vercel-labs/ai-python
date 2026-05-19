"""Tests for ai.util."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, cast

import async_solipsism  # type: ignore[import-untyped]
import pytest

from ai import util


@pytest.fixture
def event_loop_policy() -> async_solipsism.EventLoopPolicy:
    return async_solipsism.EventLoopPolicy()


async def _from_list(items: list[Any], delay: float = 0) -> AsyncIterable[Any]:
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


async def _collect(aiter: AsyncIterable[Any]) -> list[Any]:
    return [item async for item in aiter]


# -- basic behavior --------------------------------------------------------


async def test_single_iterable() -> None:
    result = await _collect(util.merge(_from_list([1, 2, 3])))
    assert result == [1, 2, 3]


async def test_empty_iterable() -> None:
    result = await _collect(util.merge(_from_list([])))
    assert result == []


async def test_no_iterables() -> None:
    result = await _collect(util.merge())
    assert result == []


async def test_multiple_iterables_all_items_yielded() -> None:
    result = await _collect(
        util.merge(
            _from_list(["a", "b"]),
            _from_list(["x", "y"]),
        )
    )
    assert sorted(result) == ["a", "b", "x", "y"]


async def test_different_lengths() -> None:
    result = await _collect(
        util.merge(
            _from_list([1, 2, 3]),
            _from_list([10]),
        )
    )
    assert sorted(result) == [1, 2, 3, 10]


async def test_many_iterables() -> None:
    result = await _collect(
        util.merge(
            _from_list([1]),
            _from_list([2]),
            _from_list([3]),
            _from_list([4]),
        )
    )
    assert sorted(result) == [1, 2, 3, 4]


# -- timing (async-solipsism) ---------------------------------------------


async def test_simulated_clock_advances() -> None:
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await _collect(
        util.merge(
            _from_list([1, 2], delay=10),
            _from_list([3, 4], delay=5),
        )
    )
    elapsed = loop.time() - t0
    assert elapsed == pytest.approx(20.0)


async def test_ordering_shorter_delay_first() -> None:
    result = await _collect(
        util.merge(
            _from_list(["slow"], delay=100),
            _from_list(["fast"], delay=1),
        )
    )
    assert result == ["fast", "slow"]


# -- error handling --------------------------------------------------------


async def test_error_cancels_other_iterables() -> None:
    """When one iterable raises, the others are closed."""
    closed: list[str] = []

    async def good() -> AsyncIterable[str]:
        try:
            await asyncio.sleep(10)
            yield "never"
        finally:
            closed.append("good")

    async def bad() -> AsyncIterable[str]:
        await asyncio.sleep(1)
        raise RuntimeError("boom")
        yield "unreachable"

    with pytest.raises(ExceptionGroup) as exc_info:
        await _collect(util.merge(good(), bad()))

    assert exc_info.group_contains(RuntimeError, match="boom")
    assert "good" in closed


async def test_error_propagates() -> None:
    """The original exception is re-raised after cleanup."""

    async def failing() -> AsyncIterable[int]:
        yield 1
        raise ValueError("oops")

    with pytest.raises(ExceptionGroup) as exc_info:
        await _collect(util.merge(failing()))

    assert exc_info.group_contains(ValueError, match="oops")


async def test_items_before_error_are_yielded() -> None:
    """Items yielded before the error are still collected."""

    async def ok() -> AsyncIterable[str]:
        yield "a"
        await asyncio.sleep(100)
        yield "b"

    async def fails_later() -> AsyncIterable[str]:
        yield "x"
        await asyncio.sleep(1)
        raise RuntimeError("fail")

    results: list[str] = []
    with pytest.raises(ExceptionGroup) as exc_info:
        async for item in util.merge(ok(), fails_later()):
            results.append(item)

    assert exc_info.group_contains(RuntimeError, match="fail")
    assert "x" in results


async def test_cleanup_with_non_generator_iterable() -> None:
    """Iterables without aclose are handled gracefully."""

    class SimpleIter:
        def __init__(self) -> None:
            self.values = iter([1, 2])

        def __aiter__(self) -> SimpleIter:
            return self

        async def __anext__(self) -> int:
            try:
                return next(self.values)
            except StopIteration:
                raise StopAsyncIteration from None

    async def failing() -> AsyncIterable[int]:
        raise RuntimeError("boom")
        yield 0

    with pytest.raises(ExceptionGroup) as exc_info:
        await _collect(util.merge(SimpleIter(), failing()))

    assert exc_info.group_contains(RuntimeError, match="boom")


# -- unwrap_generator_exit --------------------------------------------------


async def test_unwrap_generator_exit_pure_generator_exit() -> None:
    """A group containing only GeneratorExit unwraps to GeneratorExit."""
    with pytest.raises(GeneratorExit):
        async with util.unwrap_generator_exit():
            raise BaseExceptionGroup("group", [GeneratorExit()])


async def test_unwrap_generator_exit_nested_generator_exits() -> None:
    """Nested groups containing only GeneratorExits also unwrap."""
    with pytest.raises(GeneratorExit):
        async with util.unwrap_generator_exit():
            raise BaseExceptionGroup(
                "outer",
                [BaseExceptionGroup("inner", [GeneratorExit()])],
            )


async def test_unwrap_generator_exit_mixed_propagates() -> None:
    """A group with non-GeneratorExit exceptions propagates as-is."""
    with pytest.raises(BaseExceptionGroup) as exc_info:
        async with util.unwrap_generator_exit():
            raise BaseExceptionGroup(
                "group", [GeneratorExit(), ValueError("x")]
            )
    assert exc_info.group_contains(ValueError, match="x")


async def test_unwrap_generator_exit_non_group_passes_through() -> None:
    """Non-group exceptions pass through unchanged."""
    with pytest.raises(ValueError, match="x"):
        async with util.unwrap_generator_exit():
            raise ValueError("x")


async def test_unwrap_generator_exit_no_exception() -> None:
    """No exception → context manager returns normally."""
    async with util.unwrap_generator_exit():
        pass


# -- maybe_aclosing --------------------------------------------------------


async def test_maybe_aclosing_calls_aclose_on_asyncgen() -> None:
    """A normal async generator's aclose runs when the with block exits."""
    closed = False

    async def gen() -> AsyncIterator[int]:
        nonlocal closed
        try:
            yield 1
        finally:
            closed = True

    async with util.maybe_aclosing(gen()) as g:
        async for _ in g:
            break
    assert closed


async def test_maybe_aclosing_no_aclose_attribute() -> None:
    """Iterables without an aclose method are handled gracefully."""

    class SimpleIter:
        def __aiter__(self) -> SimpleIter:
            return self

        async def __anext__(self) -> int:
            raise StopAsyncIteration

    s = SimpleIter()
    async with util.maybe_aclosing(s) as it:
        assert it is s


async def test_maybe_aclosing_runs_aclose_on_exception() -> None:
    """aclose still runs when the body raises."""
    closed = False

    async def gen() -> AsyncIterator[int]:
        nonlocal closed
        try:
            yield 1
        finally:
            closed = True

    with pytest.raises(ValueError):
        async with util.maybe_aclosing(gen()) as g:
            async for _ in g:
                raise ValueError
    assert closed


# -- decouple --------------------------------------------------------------


async def test_decouple_yields_all_items() -> None:
    """Basic: every item from the source is yielded in order."""
    result = await _collect(
        util.decouple(_from_list([1, 2, 3]), task_group=None)
    )
    assert result == [1, 2, 3]


async def test_decouple_with_task_group() -> None:
    """Works equivalently when given an explicit TaskGroup."""

    async def consume() -> list[int]:
        async with asyncio.TaskGroup() as tg:
            return await _collect(
                util.decouple(_from_list([1, 2, 3]), task_group=tg)
            )

    assert await consume() == [1, 2, 3]


async def test_decouple_forwards_exception_to_consumer() -> None:
    """An exception raised by the source surfaces in the consumer."""

    async def failing() -> AsyncIterable[int]:
        yield 1
        raise ValueError("boom")

    items: list[int] = []
    with pytest.raises(ValueError, match="boom"):
        async for x in util.decouple(failing(), task_group=None):
            items.append(x)
    assert items == [1]


async def test_decouple_contextvar_stable_across_yields() -> None:
    """ContextVars set in the source persist across decouple yields."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar("test")

    async def src() -> AsyncIterator[str]:
        var.set("hello")
        yield "a"
        # If the next anext ran in a different task context, this would
        # fall back to the default and the lookup would raise.
        assert var.get() == "hello"
        yield "b"

    assert await _collect(util.decouple(src(), task_group=None)) == ["a", "b"]


async def test_decouple_aclose_runs_iter_cleanup_in_worker_context() -> None:
    """Breaking the consumer aclose's the source in the worker's task context.

    foo's finally calls ``var.reset(token)``, which raises ``ValueError`` if
    the reset runs in a different context than the matching ``set``. So if
    decouple drives the source's cleanup from a foreign task, this test
    surfaces it as the reset raising.
    """
    var: contextvars.ContextVar[str] = contextvars.ContextVar("test")
    cleanup_seen: str | None = None

    async def src() -> AsyncIterator[int]:
        token = var.set("worker")
        try:
            for i in range(100):
                yield i
        finally:
            nonlocal cleanup_seen
            cleanup_seen = var.get()
            var.reset(token)  # would raise on context mismatch

    n = 0
    async with util.maybe_aclosing(util.decouple(src(), task_group=None)) as it:
        async for _ in it:
            n += 1
            if n == 3:
                break

    assert cleanup_seen == "worker"


# -- merge: TaskGroup-inside-asyncgen wrapping ----------------------------


async def test_merge_aclose_returns_cleanly_after_break() -> None:
    """Aclose'ing merge mid-stream returns without raising BaseExceptionGroup.

    Without ``unwrap_generator_exit``, the TaskGroup's __aexit__ wraps the
    GeneratorExit thrown by aclose into a BaseExceptionGroup, which then
    propagates out of aclose itself.
    """

    async def src() -> AsyncIterator[int]:
        for i in range(100):
            yield i

    m = util.merge(src())
    n = 0
    async for _ in m:
        n += 1
        if n == 3:
            break

    # Should complete without raising BaseExceptionGroup.
    await cast(Any, m).aclose()


async def test_merge_preserves_contextvar_across_yields() -> None:
    """merge must keep contextvars consistent across yields from the same iter.

    On main, ``merge`` wraps each ``__anext__`` call in a fresh task, so a
    contextvar set in one anext is invisible in the next, and a finally-block
    ``var.reset(token)`` raises ValueError because the token was created in
    a different task's context than the one running reset.
    """
    var: contextvars.ContextVar[str] = contextvars.ContextVar("test")

    async def src() -> AsyncIterator[int]:
        token = var.set("hello")
        try:
            yield 1
            yield 2
        finally:
            var.reset(token)  # raises ValueError on context mismatch

    items = [x async for x in util.merge(src())]
    assert items == [1, 2]


# -- merge: restart --------------------------------------------------------


class _Restartable:
    """An iterable returning a fresh async generator each call.

    Items can be queued via ``push``; each iteration drains the queue and stops.
    Tracks how many times ``__aiter__`` has been called.
    """

    def __init__(self) -> None:
        self._items: list[Any] = []
        self.iter_count = 0

    def push(self, *items: Any) -> None:
        self._items.extend(items)

    def __aiter__(self) -> AsyncIterator[Any]:
        self.iter_count += 1
        items = list(self._items)
        self._items.clear()

        async def gen() -> AsyncIterator[Any]:
            for x in items:
                yield x

        return gen()


async def test_merge_restarts_restartable_iterable() -> None:
    """A restartable iterable is re-iterated when another iterable yields."""
    src = _Restartable()
    src.push("r1")

    async def driver() -> AsyncIterator[str]:
        await asyncio.sleep(10)
        src.push("r2")
        yield "d1"
        await asyncio.sleep(10)
        src.push("r3", "r4")
        yield "d2"
        await asyncio.sleep(10)
        yield "d3"

    result = await _collect(util.merge(driver(), src))
    assert sorted(result) == ["d1", "d2", "d3", "r1", "r2", "r3", "r4"]
    # __aiter__ called once initially + once after each driver yield.
    assert src.iter_count == 4


async def test_merge_does_not_restart_async_generator() -> None:
    """A bare async generator (its own iterator) is not re-iterated."""
    runs = 0

    async def gen() -> AsyncIterator[str]:
        nonlocal runs
        runs += 1
        yield "g"

    async def driver() -> AsyncIterator[str]:
        await asyncio.sleep(10)
        yield "d1"
        await asyncio.sleep(10)
        yield "d2"

    result = await _collect(util.merge(driver(), gen()))
    assert sorted(result) == ["d1", "d2", "g"]
    assert runs == 1


async def test_merge_restart_false_disables_restart() -> None:
    """``restart=False`` prevents re-iterating restartable iterables."""
    src = _Restartable()
    src.push("r1")

    async def driver() -> AsyncIterator[str]:
        await asyncio.sleep(10)
        src.push("never1")
        yield "d1"
        await asyncio.sleep(10)
        src.push("never2")
        yield "d2"

    result = await _collect(util.merge(driver(), src, restart=False))
    assert sorted(result) == ["d1", "d2", "r1"]
    assert src.iter_count == 1


async def test_merge_restart_with_no_new_items_terminates() -> None:
    """A restart with nothing to yield doesn't cause merge to loop forever."""
    src = _Restartable()
    src.push("only")

    async def driver() -> AsyncIterator[str]:
        await asyncio.sleep(10)
        yield "d1"
        await asyncio.sleep(10)
        yield "d2"

    result = await _collect(util.merge(driver(), src))
    assert sorted(result) == ["d1", "d2", "only"]
    # Still re-iterated once per driver yield, even though nothing new arrived.
    assert src.iter_count == 3


async def test_merge_restart_with_multiple_restartables() -> None:
    """Multiple restartable iterables are each re-iterated when others fire."""
    a = _Restartable()
    b = _Restartable()
    a.push("a1")
    b.push("b1")

    async def driver() -> AsyncIterator[str]:
        await asyncio.sleep(10)
        a.push("a2")
        b.push("b2")
        yield "d1"

    result = await _collect(util.merge(driver(), a, b))
    assert sorted(result) == ["a1", "a2", "b1", "b2", "d1"]
    assert a.iter_count == 2
    assert b.iter_count == 2


async def test_merge_restart_only_after_other_iterable_yields() -> None:
    """Restart is triggered by another iterable, not self-completion."""
    src = _Restartable()
    src.push("r1")

    # Single-iterable merge: src exhausts itself and merge ends without
    # __aiter__ being called again.
    result = await _collect(util.merge(src))
    assert result == ["r1"]
    assert src.iter_count == 1


async def test_merge_restart_when_yield_and_stop_collide() -> None:
    """Restart still fires when a yield and a restartable's exhaustion land
    in the same ``asyncio.wait`` step.

    If merge processes the yielding task before the stopping one inside the
    same ``done`` set, the stopping iter's slot is still its original task
    (not ``None``) at the moment a too-eager restart pass runs — so the
    restartable never gets re-iterated to pick up items pushed by the
    consumer in response to the yield.

    ``done`` is a ``set`` so its iteration order is hash-driven (by task
    ``id``); we can't force the bad order portably, so the scenario runs
    many times to make the bad order overwhelmingly likely. Empirically
    each iteration fails ~50% of the time with the bug, so 25 iterations
    gives a miss-rate of ~10⁻⁸.
    """

    class _DelayedEmpty:
        """A restartable whose ``__aiter__`` always sleeps before yielding.

        With the same delay as the driver's yield, its initial-empty anext
        completes _at the same simulated time_ as the driver's item, putting
        both in the same ``done`` set.
        """

        def __init__(self) -> None:
            self.iter_count = 0
            self._items: list[Any] = []

        def push(self, *items: Any) -> None:
            self._items.extend(items)

        def __aiter__(self) -> AsyncIterator[Any]:
            self.iter_count += 1
            items = list(self._items)
            self._items.clear()

            async def gen() -> AsyncIterator[Any]:
                await asyncio.sleep(10)
                for x in items:
                    yield x

            return gen()

    async def one_run() -> list[str]:
        src = _DelayedEmpty()  # initially empty: first iter yields nothing

        async def driver() -> AsyncIterator[str]:
            await asyncio.sleep(10)
            yield "d1"

        results: list[str] = []
        async for item in util.merge(driver(), src):
            results.append(item)
            if item == "d1":
                src.push("r1")
        return results

    for _ in range(25):
        results = await one_run()
        assert sorted(results) == ["d1", "r1"], results


def test_merge_cleanup_on_asyncio_shutdown() -> None:
    """A partially consumed merge gen is cleaned up on shutdown.

    The consumer breaks out of ``async for x in merge(src())`` without
    explicitly aclose'ing the merge gen, so cleanup is left to asyncio.run's
    shutdown sequence (``_cancel_all_tasks`` then ``shutdown_asyncgens``).
    The source's finally must still run in a context that matches its
    matched ``var.set`` — i.e. the cancellation has to drive ``src``'s
    cleanup from the same task that called ``var.set``. Run as a sync test
    so we drive a real ``asyncio.run`` lifecycle instead of pytest-asyncio's
    per-test loop.
    """
    cleanup_log: list[str] = []
    var: contextvars.ContextVar[str] = contextvars.ContextVar("test")

    async def src() -> AsyncIterator[int]:
        token = var.set("hello")
        try:
            for i in range(100):
                yield i
        finally:
            try:
                var.reset(token)
                cleanup_log.append("ok")
            except Exception as e:
                cleanup_log.append(f"err:{type(e).__name__}")

    async def main() -> None:
        async for x in util.merge(src()):
            if x == 3:
                break
        # No explicit aclose; rely on asyncio.run shutdown.

    asyncio.run(main())

    assert cleanup_log == ["ok"], cleanup_log
