from __future__ import annotations

import asyncio
import contextlib
import functools
import gc
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from typing import cast

import anyio
import pydantic
import pytest

from ai.contrib.sandbox import runner as runner_

# Runner-side state, read back with ``w.run``.
_produced = [0]
_closed: list[str] = []


def gen(n: int, step: int = 1) -> Generator[int, None, str]:
    yield from range(0, n, step)
    return "rv"  # noqa: B901


async def agen(n: int, delay: float = 0.0) -> AsyncGenerator[int]:
    for i in range(n):
        await asyncio.sleep(delay)
        yield i


def boom() -> Generator[int]:
    yield 1
    raise ValueError("boom")


def endless() -> Generator[int]:
    try:
        i = 0
        while True:
            yield i
            i += 1
    finally:
        _closed.append("closed")


async def stuck() -> AsyncGenerator[int]:
    try:
        yield 1
    finally:
        await asyncio.sleep(3600)


def fast(n: int) -> Generator[int]:
    for i in range(n):
        _produced[0] += 1
        yield i


class Point(pydantic.BaseModel):
    x: int
    y: int


class UnbuiltinError(Exception):
    pass


def point() -> Point:
    return Point(x=1, y=2)


def unbuiltin() -> None:
    raise UnbuiltinError("nope")


def big(n: int) -> str:
    return "x" * n


def produced() -> int:
    return _produced[0]


def closed() -> list[str]:
    return _closed


def reset() -> None:
    _produced[0] = 0
    _closed.clear()


# Not a fixture: the runner has to be closed in the task that opened it.
@contextlib.asynccontextmanager
async def _runner() -> AsyncIterator[runner_.Runner]:
    async with runner_.local_runner(main_path=__file__) as w:
        await w.run(reset)
        yield w


async def test_run() -> None:
    async with _runner() as w:
        assert await w.run(functools.partial(int, base=16), "ff") == 255
        assert await w.run(asyncio.sleep, 0, "slept") == "slept"
        with pytest.raises(ZeroDivisionError):
            await w.run(divmod, 1, 0)


async def test_results_are_json() -> None:
    async with _runner() as w:
        result = await w.run(point)
        assert result == {"x": 1, "y": 2}
        assert Point.model_validate(result) == point()
        async with w.iterate(map, str, [1, 2]) as it:
            assert [x async for x in it] == ["1", "2"]
        # Not JSON, and it wouldn't be unpickled here if it were sent.
        with pytest.raises(runner_.RemoteError, match="serialize"):
            await w.run(object)
        assert await w.run(int, "7") == 7


async def test_errors() -> None:
    async with _runner() as w:
        with pytest.raises(
            runner_.RemoteError, match="UnbuiltinError: nope"
        ) as e:
            await w.run(unbuiltin)
        assert "in unbuiltin" in e.value.__notes__[0]
        with pytest.raises(ZeroDivisionError) as builtin:
            await w.run(divmod, 1, 0)
        assert "Remote traceback" in builtin.value.__notes__[0]


@pytest.mark.parametrize("free_running", [False, True])
async def test_iterate(free_running: bool) -> None:
    async with _runner() as w:
        async with w.iterate(gen, 5, free_running=free_running) as it:
            assert [x async for x in it] == [0, 1, 2, 3, 4]
        async with w.iterate(agen, 4, free_running=free_running) as it:
            assert [x async for x in it] == [0, 1, 2, 3]
        with pytest.raises(ValueError, match="boom"):
            async with w.iterate(boom, free_running=free_running) as it:
                async for _ in it:
                    pass


@pytest.mark.parametrize("free_running", [False, True])
async def test_close_early(free_running: bool) -> None:
    async with _runner() as w:
        async with w.iterate(endless, free_running=free_running) as it:
            async for x in it:
                if x == 3:
                    break
        with anyio.fail_after(5):
            while not await w.run(closed):
                await anyio.sleep(0.01)


@pytest.mark.parametrize("free_running", [False, True])
async def test_abandoned(free_running: bool) -> None:
    async with _runner() as w:
        it = w.iterate(endless, free_running=free_running)
        await it.__anext__()
        del it
        gc.collect()
        # Abandoned iterators are closed along with the next request.
        with anyio.fail_after(5):
            while not await w.run(closed):
                await anyio.sleep(0.01)


async def test_close_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_, "_CLOSE_TIMEOUT", 0.1)
    async with _runner() as w:
        it = w.iterate(stuck)
        assert await it.__anext__() == 1
        with anyio.fail_after(5), pytest.raises(TimeoutError):
            await it.aclose()
        assert await w.run(int, "7") == 7


async def test_cancel_while_waiting() -> None:
    async with _runner() as w:
        with anyio.move_on_after(0.1):
            async with w.iterate(agen, 100, 1.0, free_running=True) as it:
                async for _ in it:
                    pass
        assert await w.run(int, "7") == 7


async def test_free_running_doesnt_wait_to_be_asked() -> None:
    async with _runner() as w:
        # Asked for one at a time, the rest would take at least 0.2s.
        async with w.iterate(agen, 5, 0.05, free_running=True) as it:
            assert await it.__anext__() == 0
            await anyio.sleep(0.5)
            with anyio.fail_after(0.1):
                assert [x async for x in it] == [1, 2, 3, 4]


@pytest.mark.parametrize("window", [1, 2, 4, 7])
async def test_window(window: int) -> None:
    async with _runner() as w:
        seen = []
        async with w.iterate(fast, 200, free_running=True, window=window) as it:
            async for x in it:
                seen.append(x)
                if len(seen) in (1, 50):
                    await anyio.sleep(0.1)
                    assert await w.run(produced) - len(seen) <= window
        assert seen == list(range(200))


def test_window_must_be_positive() -> None:
    w = runner_.local_runner()
    with pytest.raises(ValueError, match="window"):
        w.iterate(fast, 1, free_running=True, window=0)


@contextlib.asynccontextmanager
async def _scripted_process(
    output: list[bytes],
) -> AsyncIterator[runner_.Process]:
    """A runner that answers its first request with ``output``, all at once."""
    send, receive = anyio.create_memory_object_stream[bytes](len(output))

    class Stdin:
        async def send(self, data: bytes) -> None:
            while output:
                send.send_nowait(output.pop(0))

    class Scripted:
        stdin = Stdin()
        stdout = receive

        async def exit_message(self) -> str:
            await anyio.sleep_forever()
            raise AssertionError

    try:
        yield cast("runner_.Process", Scripted())
    finally:
        send.close()
        receive.close()


async def test_items_beyond_credit_fail_the_stream() -> None:
    window = 4
    flood = [runner_.encode_reply(0, "ok", ["yield", i]) for i in range(100)]
    async with runner_.Runner(_scripted_process(flood)) as w:
        seen = []
        async with w.iterate(gen, 0, free_running=True, window=window) as it:
            with pytest.raises(RuntimeError, match="credit"):
                async for x in it:
                    seen.append(x)
                    await anyio.sleep(0.01)
        assert len(seen) <= window + 3
        assert not w._streams


async def test_oversized_reply_closes_the_runner() -> None:
    reply = runner_.encode_reply(0, "ok", "x" * 100)
    process = _scripted_process([reply])
    async with runner_.Runner(process, max_reply_bytes=50) as w:
        for _ in range(2):
            with pytest.raises(ValueError, match="limit is 50"):
                await w.run(produced)


async def test_stuck_runner_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_, "_CLOSE_TIMEOUT", 0.5)
    with anyio.fail_after(10):
        async with runner_.local_runner(
            main_path=__file__, max_reply_bytes=50
        ) as w:
            with pytest.raises(ValueError, match="limit is 50"):
                await w.run(big, 10_000_000)
