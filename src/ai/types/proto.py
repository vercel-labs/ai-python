from typing import Any, Protocol, runtime_checkable

from . import tools


@runtime_checkable
class ToolSchemaLike(Protocol):
    """Anything that exposes a tool schema to the LLM layer.

    Structural type: ``name``, ``description``, ``param_schema``.
    Satisfied by both :class:`ToolSchema` and the agents' ``Tool`` class.
    """

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def param_schema(self) -> dict[str, Any]: ...


# Anything the LLM layer can use as a tool: either a host-executed
# function tool (described by a schema) or a provider-executed built-in
# tool subclass.
type ToolLike = ToolSchemaLike | tools.BuiltinTool
