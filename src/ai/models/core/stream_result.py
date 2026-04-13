"""StreamResult — concrete wrapper around a message stream."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ...types import messages as messages_


class StreamResult:
    """Wrapper around a message stream. Async-iterable; collects the final result.

    Properties like ``.text`` and ``.tool_calls`` delegate to the final
    ``Message`` snapshot and are available after iteration completes.

    Satisfies :class:`~ai.types.StreamResultLike`.
    """

    def __init__(self, gen: AsyncGenerator[messages_.Message]) -> None:
        self._gen = gen
        self._final: messages_.Message | None = None

    @classmethod
    def from_generator(cls, gen: AsyncGenerator[messages_.Message]) -> StreamResult:
        """Create a :class:`StreamResult` from an async generator.

        This is the public API for middleware that needs to transform or
        replace the stream returned by ``wrap_model``::

            async def wrap_model(self, call, next):
                original = await next(call)

                async def _transformed():
                    async for msg in original:
                        yield modify(msg)

                return StreamResult.from_generator(_transformed())
        """
        return cls(gen)

    def __aiter__(self) -> AsyncGenerator[messages_.Message]:
        return self._iterate()

    async def _iterate(self) -> AsyncGenerator[messages_.Message]:
        async for msg in self._gen:
            self._final = msg
            yield msg

    @property
    def text(self) -> str:
        return self._final.text if self._final else ""

    @property
    def tool_calls(self) -> list[messages_.ToolCallPart]:
        return self._final.tool_calls if self._final else []

    @property
    def usage(self) -> messages_.Usage | None:
        return self._final.usage if self._final else None

    @property
    def output(self) -> Any:
        """Parsed structured output from the final message, if available."""
        return self._final.output if self._final else None
