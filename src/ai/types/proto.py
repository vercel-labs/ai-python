from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

from . import messages, usage


@runtime_checkable
class ToolLike(Protocol):
    """Anything the LLM layer can use as a tool definition."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def param_schema(self) -> dict[str, Any]: ...


@runtime_checkable
class StreamResultLike(Protocol):
    """Structural protocol satisfied by :class:`ai.models.StreamResult`.

    Middleware that transforms or replaces the stream returned by
    ``wrap_model`` should return an object satisfying this protocol.
    The easiest way is ``StreamResult.from_generator(gen)``.
    """

    def __aiter__(self) -> AsyncGenerator[messages.Message]: ...

    @property
    def text(self) -> str: ...

    @property
    def tool_calls(self) -> list[messages.ToolCallPart]: ...

    @property
    def usage(self) -> usage.Usage | None: ...

    @property
    def output(self) -> Any: ...

    @property
    def turn_id(self) -> str | None: ...
