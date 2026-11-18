"""The protocols a runner implements, and the ambient runner.

See :mod:`ai.remote`.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
)
from typing import Any, Protocol

__all__ = ["RemoteIterator", "Runner", "use_runner"]


class RemoteIterator(Protocol):
    """An async iterator over a remote iterable; ``async with`` closes it."""

    def __aiter__(self) -> AsyncIterator[Any]: ...

    def __anext__(self) -> Awaitable[Any]: ...

    async def __aenter__(self) -> AsyncIterator[Any]: ...

    async def __aexit__(self, *exc_info: object) -> None: ...


class Runner(Protocol):
    """Runs functions somewhere else.

    Arguments and the functions themselves are pickled.  Only
    positional arguments are passed through; use ``functools.partial``
    for keywords.  With ``free_running``, ``iterate`` sends items as
    they come rather than waiting to be asked for each; ``ai.remote``
    always asks for that.

    The runner may be running code that isn't trusted, so nothing it
    sends is unpickled: results and items come back as JSON data, to be
    validated into the type the caller expects.
    """

    async def run[*Ts](
        self, func: Callable[[*Ts], object], *args: *Ts
    ) -> Any: ...

    def iterate[*Ts](
        self,
        func: Callable[[*Ts], Iterable[object] | AsyncIterable[object]],
        *args: *Ts,
        free_running: bool = False,
    ) -> RemoteIterator: ...


_runner: contextvars.ContextVar[Runner | None] = contextvars.ContextVar(
    "ai_runner", default=None
)

# Set in the runner, where everything runs in place.
_in_runner: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ai_in_runner", default=False
)


@contextlib.asynccontextmanager
async def use_runner(runner: Runner) -> AsyncIterator[None]:
    """Make *runner* the ambient runner until exit.

    This doesn't start or stop it.
    """
    token = _runner.set(runner)
    try:
        yield
    finally:
        _runner.reset(token)
