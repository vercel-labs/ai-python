from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolLike(Protocol):
    """Anything the LLM layer can use as a tool definition."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def param_schema(self) -> dict[str, Any]: ...
