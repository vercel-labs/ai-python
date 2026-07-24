"""Google provider-executed tools."""

from __future__ import annotations

from ... import types


def google_search() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="google_search",
        tool_config=types.tools.ToolConfig(id="google.google_search"),
    )


def url_context() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="url_context",
        tool_config=types.tools.ToolConfig(id="google.url_context"),
    )


def code_execution() -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="code_execution",
        tool_config=types.tools.ToolConfig(id="google.code_execution"),
    )


__all__ = ["code_execution", "google_search", "url_context"]
