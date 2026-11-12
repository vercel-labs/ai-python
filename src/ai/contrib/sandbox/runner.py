"""Run functions in another Python process.

``python -m ai.contrib.sandbox.runner`` runs a runner, and ``Runner``
talks to one, started by ``local_runner`` or ``ai.contrib.sandbox.vercel``.

The runner reads requests from stdin and writes replies to stdout. Each request
is a pickled ``(request_id, command)`` pair, with the command pickled
separately inside it, so a command that fails to unpickle can still be
answered. Commands:

* ``("init", main_path)`` loads ``main_path`` as the ``__main__`` module, so
  functions pickled by reference to ``__main__`` resolve.
* ``("run", func, args)`` calls ``func(*args)``. If that
  returns an awaitable, such as from a coroutine function, it runs as a task on
  the runner's event loop, concurrently with other requests.
* ``("iter", id, func, args)`` calls ``func(*args)`` and
  keeps the result, a sync or async generator or any iterable, under ``id``.
* ``("next", id)`` advances iterator ``id``, replying ``("yield", value)`` or,
  once it is exhausted, ``("done", return_value)``.
* ``("close", id)`` closes iterator ``id`` early.
* ``("stream", func, args, window)`` iterates over ``func(*args)`` without
  waiting to be asked, replying ``("yield", value)`` for each item and then
  ``("done", return_value)``, all under the one request id. Each item uses up
  a credit, of which there are ``window`` to start with; with none left, the
  iterator isn't advanced until more arrive. Cancelling the request closes the
  iterator.
* ``("credit", request_id, n)`` gives stream ``request_id`` ``n`` more credits.
* ``("cancel", request_id)`` cancels that request's task, if it is still
  running. Cancelled requests get no reply.

Sync functions run on the event loop's thread, so they run one at a time and
block other requests while they do. They also run in the runner's root
context, so context variables they set are seen by every later request; async
functions run in a copy of it.

Every other request but ``stream`` gets exactly one reply, the JSON
``[request_id, "ok", value]`` or ``[request_id, "err", error]``, where
``error`` has the exception's ``type``, ``message`` and ``traceback``. Replies
arrive as requests finish, in any order. Each is framed as an 8-byte big-endian
length followed by the JSON.

Replies are JSON and not pickles because the runner may be running code that
isn't trusted, and unpickling runs code. So results come back as plain JSON
data, to be validated into whatever type the caller expects, and an exception
comes back as a builtin exception of the same name if there is one and as a
``RemoteError`` otherwise.
"""

from __future__ import annotations

import asyncio
import builtins
import dataclasses
import inspect
import io
import itertools
import json
import math
import os
import pickle
import runpy
import struct
import sys
import threading
import traceback
import types
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from typing import IO, TYPE_CHECKING, Any, Protocol, Self

import anyio
import anyio.abc
import pydantic_core
from anyio.streams.buffered import BufferedByteReceiveStream

from .util import protected

if TYPE_CHECKING:
    from anyio.abc import ByteReceiveStream, ByteSendStream
    from anyio.streams.memory import (
        MemoryObjectReceiveStream,
        MemoryObjectSendStream,
    )

_LENGTH = struct.Struct(">Q")

DEFAULT_WINDOW = 256


class RemoteError(Exception):
    """An exception raised in the runner that has no builtin type."""


def encode_reply(request_id: int, status: str, value: Any) -> bytes:
    """Frame a reply; raises if ``value`` can't be serialized."""
    data = pydantic_core.to_json([request_id, status, value])
    return _LENGTH.pack(len(data)) + data


def _error(exc: BaseException) -> dict[str, str]:
    cls = type(exc)
    name = cls.__qualname__
    if cls.__module__ != "builtins":
        name = f"{cls.__module__}.{name}"
    return {
        "type": name,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def unwrap_reply(reply: tuple[str, Any]) -> Any:
    status, value = reply
    if status != "err":
        return value
    name, message = str(value["type"]), str(value["message"])
    exc: Exception = RemoteError(f"{name}: {message}")
    cls = getattr(builtins, name, None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            exc = cls(message)
        except Exception:
            pass
    exc.add_note(f"Remote traceback:\n{value['traceback']}")
    raise exc


def _load_main(path: str) -> None:
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    # Same approach as multiprocessing's spawn start method: run the script
    # under a name that skips its `if __name__ == "__main__"` block.
    main = types.ModuleType("__mp_main__")
    sys.modules["__main__"] = sys.modules["__mp_main__"] = main
    main.__dict__.update(runpy.run_path(path, run_name="__mp_main__"))


class _Server:
    """Handles requests on the runner's event loop."""

    def __init__(self, out: IO[bytes]) -> None:
        self._out = out
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._iterators: dict[int, Iterator[Any] | AsyncIterator[Any]] = {}
        self._credits: dict[int, _Credits] = {}
        self._shutting_down = False
        task = asyncio.current_task()
        assert task is not None
        self._root_context = task.get_context()

    def _write(self, data: bytes) -> None:
        self._out.write(data)
        self._out.flush()

    def _reply(self, request_id: int, status: str, value: Any) -> None:
        if self._shutting_down:
            return
        try:
            data = encode_reply(request_id, status, value)
        except Exception as exc:
            data = encode_reply(request_id, "err", _error(exc))
        self._write(data)

    def start(self, request_id: int, payload: bytes) -> None:
        try:
            kind, value = self._handle(pickle.loads(payload))
        except Exception as exc:
            self._reply(request_id, "err", _error(exc))
            return
        match kind:
            case "value":
                self._reply(request_id, "ok", value)
            case "await":
                self._tasks[request_id] = asyncio.create_task(
                    self._finish(request_id, value)
                )
            case "stream":
                it, window = value
                credits = self._credits[request_id] = _Credits(window)
                self._tasks[request_id] = asyncio.create_task(
                    self._stream(request_id, it, credits)
                )

    async def _finish(self, request_id: int, awaitable: Awaitable[Any]) -> None:
        try:
            value = await awaitable
        except asyncio.CancelledError:
            self._reply(
                request_id,
                "err",
                _error(RuntimeError("remote call was cancelled")),
            )
        except Exception as exc:
            self._reply(request_id, "err", _error(exc))
        else:
            self._reply(request_id, "ok", value)
        finally:
            self._tasks.pop(request_id, None)

    async def _stream(
        self,
        request_id: int,
        it: Iterator[Any] | AsyncIterator[Any],
        credits: _Credits,
    ) -> None:
        try:
            try:
                await self._drain(request_id, it, credits)
            finally:
                if isinstance(it, AsyncIterator):
                    if (aclose := getattr(it, "aclose", None)) is not None:
                        await aclose()
                elif (close := getattr(it, "close", None)) is not None:
                    self._root_context.run(close)
        except asyncio.CancelledError:
            self._reply(
                request_id,
                "err",
                _error(RuntimeError("remote call was cancelled")),
            )
        except Exception as exc:
            self._reply(request_id, "err", _error(exc))
        else:
            self._reply(request_id, "ok", ("done", None))
        finally:
            self._tasks.pop(request_id, None)
            self._credits.pop(request_id, None)

    async def _drain(
        self,
        request_id: int,
        it: Iterator[Any] | AsyncIterator[Any],
        credits: _Credits,
    ) -> None:
        """Send each item of ``it``.

        Sync iterators advance in the root context, as in ``next``.  An
        item that fails to serialize raises, ending the stream.
        """
        while True:
            await credits.take()
            if isinstance(it, AsyncIterator):
                try:
                    item = await it.__anext__()
                except StopAsyncIteration:
                    return
            else:
                try:
                    item = self._root_context.run(next, it)
                except StopIteration:
                    return
            self._write(encode_reply(request_id, "ok", ("yield", item)))
            await asyncio.sleep(0)

    async def shutdown(self) -> None:
        self._shutting_down = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _handle(self, command: tuple[Any, ...]) -> tuple[str, Any]:
        """Return what to do with ``command``.

        That's ``("value", value)`` to reply with, ``("await",
        awaitable)`` to await and reply with, or ``("stream", (iterator,
        window))`` to stream.
        """
        match command:
            case ("init", path):
                _load_main(path)
                return "value", None
            case ("run", func, args):
                result = func(*args)
                if inspect.isawaitable(result):
                    return "await", result
                return "value", result
            case ("iter", iter_id, func, args):
                self._iterators[iter_id] = _as_iterator(func(*args))
                return "value", None
            case ("stream", func, args, window):
                return "stream", (_as_iterator(func(*args)), window)
            case ("credit", target, n):
                if credits := self._credits.get(target):
                    credits.add(n)
                return "value", None
            case ("next", iter_id):
                it = self._iterators[iter_id]
                if isinstance(it, AsyncIterator):
                    return "await", self._anext(iter_id, it)
                return "value", self._next(iter_id, it)
            case ("close", iter_id):
                closing: Any = self._iterators.pop(iter_id, None)
                if hasattr(closing, "aclose"):
                    return "await", closing.aclose()
                if hasattr(closing, "close"):
                    closing.close()
                return "value", None
            case ("cancel", target):
                if task := self._tasks.get(target):
                    task.cancel()
                return "value", None
            case _:
                raise ValueError(f"unknown command: {command!r}")

    def _next(self, iter_id: int, it: Iterator[Any]) -> tuple[str, Any]:
        try:
            return "yield", next(it)
        except StopIteration:
            del self._iterators[iter_id]
            return "done", None
        except BaseException:
            del self._iterators[iter_id]
            raise

    async def _anext(
        self, iter_id: int, it: AsyncIterator[Any]
    ) -> tuple[str, Any]:
        try:
            return "yield", await it.__anext__()
        except StopAsyncIteration:
            del self._iterators[iter_id]
            return "done", None
        except BaseException:
            self._iterators.pop(iter_id, None)
            raise


class _Credits:
    """How many more items a stream may send."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._added = asyncio.Event()

    def add(self, n: int) -> None:
        self._n += n
        self._added.set()

    async def take(self) -> None:
        while self._n == 0:
            self._added.clear()
            await self._added.wait()
        self._n -= 1


def _as_iterator(
    iterable: Iterable[Any] | AsyncIterable[Any],
) -> Iterator[Any] | AsyncIterator[Any]:
    if isinstance(iterable, AsyncIterable):
        return aiter(iterable)
    return iter(iterable)


async def _serve(stdin: IO[bytes], out: IO[bytes]) -> None:
    loop = asyncio.get_running_loop()
    requests: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue()

    def read_requests() -> None:
        while True:
            try:
                request = pickle.load(stdin)
            except Exception:
                request = None
            loop.call_soon_threadsafe(requests.put_nowait, request)
            if request is None:
                return

    threading.Thread(target=read_requests, daemon=True).start()
    server = _Server(out)
    while (request := await requests.get()) is not None:
        server.start(*request)
    await server.shutdown()


def main() -> None:
    """Run a runner on this process's stdin and stdout."""
    stdin = sys.stdin.buffer
    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Anything the called functions print goes to stderr, keeping stdout
    # for replies.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    asyncio.run(_serve(stdin, out))


type _Reply = tuple[str, Any] | BaseException


class _Pending:
    def __init__(self) -> None:
        self.done = anyio.Event()
        self.reply: tuple[str, Any] | None = None
        self.error: BaseException | None = None


class Process(Protocol):
    """A running runner process: its stdin, its stdout, and how it ended.

    ``Runner`` takes an async context manager that starts one and, on
    exit, stops it.
    """

    @property
    def stdin(self) -> ByteSendStream: ...

    @property
    def stdout(self) -> ByteReceiveStream: ...

    async def exit_message(self) -> str:
        """Describe how the process ended, once its output has closed."""
        ...


class Runner:
    """A connection to a runner, used as an async context manager.

    ``process`` starts the runner process on entry and stops it on exit.
    ``main_path``, if given, is loaded as the runner's ``__main__``.
    Calls can be made concurrently from several tasks; each waits only for its
    own reply.
    """

    def __init__(
        self,
        process: AbstractAsyncContextManager[Process],
        main_path: str | None = None,
    ) -> None:
        self._process_cm = process
        self._main_path = main_path
        self._iter_ids = itertools.count()
        self._request_ids = itertools.count()
        self._pending: dict[int, _Pending] = {}
        self._streams: dict[int, MemoryObjectSendStream[_Reply]]
        self._streams = {}
        self._closed = False
        # Commands from RemoteIterator.__del__, sent with the next request.
        self._abandoned: list[tuple[Any, ...]] = []

    async def __aenter__(self) -> Self:
        async with AsyncExitStack() as stack:
            self._process = await stack.enter_async_context(self._process_cm)
            await self._start()
            stack.push_async_callback(self._stop)
            self._exit_stack = stack.pop_all()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> bool | None:
        return await self._exit_stack.__aexit__(exc_type, exc, tb)

    async def _start(self) -> None:
        """Start reading replies, then send ``init``."""
        self._stdin = self._process.stdin
        self._stdout = BufferedByteReceiveStream(self._process.stdout)
        self._send_lock = anyio.Lock()
        self._reader_scope = anyio.CancelScope()
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._read_replies)
        try:
            if self._main_path is not None:
                await self._call(("init", self._main_path))
        except BaseException:
            await self._stop()
            raise

    async def _stop(self) -> None:
        self._reader_scope.cancel()
        await self._task_group.__aexit__(None, None, None)

    async def _closed_message(self) -> str:
        return await self._process.exit_message()

    async def _read_replies(self) -> None:
        with self._reader_scope:
            await self._read_replies_until_closed()

    async def _read_replies_until_closed(self) -> None:
        try:
            while True:
                (size,) = _LENGTH.unpack(
                    await self._stdout.receive_exactly(_LENGTH.size)
                )
                request_id, status, value = json.loads(
                    await self._stdout.receive_exactly(size)
                )
                if pending := self._pending.get(request_id):
                    pending.reply = status, value
                    pending.done.set()
                elif stream := self._streams.get(request_id):
                    stream.send_nowait((status, value))
        except (
            anyio.EndOfStream,
            anyio.IncompleteRead,
            anyio.BrokenResourceError,
        ):
            error: BaseException | None = None
        except Exception as exc:
            error = exc
        self._closed = True
        if self._pending or self._streams:
            if error is None:
                error = EOFError(await self._closed_message())
            for pending in self._pending.values():
                pending.error = error
                pending.done.set()
            for stream in self._streams.values():
                stream.send_nowait(error)

    async def _send(self, request_id: int, command: tuple[Any, ...]) -> None:
        data = pickle.dumps((request_id, pickle.dumps(command)))
        async with self._send_lock:
            # Nothing waits for the replies to abandoned commands, so the
            # reader drops them.
            closes = []
            while self._abandoned:
                close = self._abandoned.pop()
                closes.append(
                    pickle.dumps((next(self._request_ids), pickle.dumps(close)))
                )
            # A send interrupted partway would leave half a request in the
            # stream.

            # XXX: but is this actually quite what we want?? We might
            # still want some kind of timeout... Hm.
            await protected(self._stdin.send(b"".join(closes) + data))

    async def _call(self, command: tuple[Any, ...]) -> Any:
        if self._closed:
            raise EOFError(await self._closed_message())
        request_id = next(self._request_ids)
        pending = self._pending[request_id] = _Pending()
        try:
            await self._send(request_id, command)
            await pending.done.wait()
        except anyio.get_cancelled_exc_class():
            if not pending.done.is_set() and not self._closed:
                # We are already cancelled, and taking the send lock is an
                # anyio checkpoint, so the whole send has to be protected.
                await protected(
                    self._send(next(self._request_ids), ("cancel", request_id))
                )
            raise
        finally:
            del self._pending[request_id]
        if pending.error is not None:
            raise pending.error
        assert pending.reply is not None
        return unwrap_reply(pending.reply)

    async def _open_stream(
        self, command: tuple[Any, ...]
    ) -> tuple[int, MemoryObjectReceiveStream[_Reply]]:
        """Send ``command`` and return its id and a stream of its replies.

        The stream also gets the error if the runner goes away.  Each
        reply is buffered until it is received, so the command should
        limit how many it gets.
        """
        if self._closed:
            raise EOFError(await self._closed_message())
        request_id = next(self._request_ids)
        send, receive = anyio.create_memory_object_stream[_Reply](math.inf)
        self._streams[request_id] = send
        try:
            await self._send(request_id, command)
        except BaseException:
            await self._close_stream(request_id, cancel=False)
            receive.close()
            raise
        return request_id, receive

    async def _close_stream(self, request_id: int, *, cancel: bool) -> None:
        """Stop taking replies to ``request_id``, cancelling it if asked."""
        send = self._streams.pop(request_id, None)
        if send is None:
            return
        send.close()
        if cancel and not self._closed:
            await protected(
                self._send(next(self._request_ids), ("cancel", request_id))
            )

    async def run[*Ts](self, func: Callable[[*Ts], object], *args: *Ts) -> Any:
        """Call ``func(*args)`` remotely and return its result, as JSON data."""
        return await self._call(("run", func, args))

    def iterate[*Ts](
        self,
        func: Callable[[*Ts], Iterable[object] | AsyncIterable[object]],
        *args: *Ts,
        free_running: bool = False,
        window: int = DEFAULT_WINDOW,
    ) -> RemoteIterator:
        """Iterate remotely over ``func(*args)``, an item per trip.

        Items come back as JSON data.

        With ``free_running``, the runner advances the iterator without
        waiting to be asked and sends each item as it comes, saving a
        round trip per item.  Up to ``window`` items not yet taken are
        buffered here; with that many, the runner waits.  Credit for
        more goes back once half the window has been taken, so a window
        that covers a round trip's worth of items never stalls.

        Use ``async with`` to close the remote iterator if you stop early;
        otherwise it stays open in the runner until the runner exits.
        """
        if window < 1:
            raise ValueError(f"window must be at least 1, not {window}")
        return RemoteIterator(
            self, func, args, free_running=free_running, window=window
        )


class RemoteIterator:
    """An async iterator over a sync or async iterable in a runner."""

    def __init__(
        self,
        runner: Runner,
        func: Callable[..., Iterable[object] | AsyncIterable[object]],
        args: tuple[Any, ...],
        *,
        free_running: bool = False,
        window: int = DEFAULT_WINDOW,
    ) -> None:
        self._runner = runner
        self._call_args = (func, args)
        self._free_running = free_running
        self._window = window
        self._taken = 0
        self._id = next(runner._iter_ids)
        self._request_id = -1
        self._replies: MemoryObjectReceiveStream[_Reply] | None
        self._replies = None
        self._started = False
        self._finished = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        if self._finished:
            raise StopAsyncIteration
        try:
            if self._free_running:
                kind, value = await self._next_streamed()
            else:
                kind, value = await self._next_stepped()
        except BaseException:
            await self.aclose()
            raise
        if kind == "done":
            self._finished = True
            raise StopAsyncIteration
        return value

    async def _next_stepped(self) -> tuple[str, Any]:
        if not self._started:
            self._started = True
            await self._runner._call(("iter", self._id, *self._call_args))
        kind, value = await self._runner._call(("next", self._id))
        return kind, value

    async def _next_streamed(self) -> tuple[str, Any]:
        if not self._started:
            self._started = True
            self._request_id, self._replies = await self._runner._open_stream(
                ("stream", *self._call_args, self._window)
            )
        assert self._replies is not None
        # Credit is sent here, not as an item is returned, so that a failed
        # send can't lose an item already received.
        if self._taken >= max(1, self._window // 2):
            credit = ("credit", self._request_id, self._taken)
            self._taken = 0
            await protected(
                self._runner._send(next(self._runner._request_ids), credit)
            )
        ended = False
        try:
            reply = await self._replies.receive()
            if isinstance(reply, BaseException):
                ended = True
                raise reply
            ended = reply[0] == "err"
            kind, value = unwrap_reply(reply)
            ended = kind == "done"
            if kind == "yield":
                self._taken += 1
            return kind, value
        finally:
            if ended:
                await self._end_stream(cancel=False)

    async def _end_stream(self, *, cancel: bool) -> None:
        if self._replies is not None:
            self._replies.close()
            await self._runner._close_stream(self._request_id, cancel=cancel)

    async def aclose(self) -> None:
        if self._started and not self._finished:
            self._finished = True
            if self._free_running:
                await self._end_stream(cancel=True)
            elif not self._runner._closed:
                await protected(self._runner._call(("close", self._id)))
        self._finished = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __del__(self) -> None:
        if not self._started or self._finished:
            return
        if not self._free_running:
            self._runner._abandoned.append(("close", self._id))
        elif send := self._runner._streams.pop(self._request_id, None):
            send.close()
            self._runner._abandoned.append(("cancel", self._request_id))
        if self._replies is not None:
            self._replies.close()


@dataclasses.dataclass
class _LocalProcess:
    stdin: ByteSendStream
    stdout: ByteReceiveStream
    proc: anyio.abc.Process

    async def exit_message(self) -> str:
        return f"runner exited with {await self.proc.wait()}"


@asynccontextmanager
async def local_process(
    python: str = sys.executable,
) -> AsyncIterator[Process]:
    """Start a runner subprocess on this machine."""
    async with await anyio.open_process(
        [python, "-m", _MODULE], stderr=None
    ) as proc:
        assert proc.stdin is not None and proc.stdout is not None
        yield _LocalProcess(proc.stdin, proc.stdout, proc)


def local_runner(
    main_path: str | None = None, *, python: str = sys.executable
) -> Runner:
    """Make a runner subprocess on this machine.

    ``main_path`` defaults to this process's ``__main__``.
    """
    if main_path is None:
        main_path = getattr(sys.modules["__main__"], "__file__", None)
    return Runner(local_process(python), main_path)


# This module's name even when it runs as ``__main__``.
_MODULE = "ai.contrib.sandbox.runner"

if __name__ == "__main__":
    main()
