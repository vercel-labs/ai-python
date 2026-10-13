"""MCP client: tool conversion, end-to-end execution."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
from typing import Any

import mcp.types
import pytest
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

import ai
from ai.agents.mcp.client import _mcp_tool_to_native

from ...conftest import (
    MOCK_MODEL,
    collect_messages,
    mock_llm,
    text_msg,
    tool_call_msg,
)


def _fake_mcp_tool(
    name: str = "mcp_echo", description: str = "Echo input"
) -> mcp.types.Tool:
    """Build a minimal mcp.types.Tool for testing."""
    return mcp.types.Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def _noop_transport_factory() -> contextlib.AbstractAsyncContextManager[Any]:
    """Dummy transport factory — never actually called in these tests."""
    raise NotImplementedError("should not be called")


# -- _mcp_tool_to_native produces a valid Tool ----------------------------


def test_mcp_tool_to_native_basic() -> None:
    """Converting an MCP tool to native produces a Tool with correct schema."""
    mcp_tool = _fake_mcp_tool(name="mcp_basic_test")
    native = _mcp_tool_to_native(
        mcp_tool, "test:key", _noop_transport_factory, None
    )

    assert native.name == "mcp_basic_test"
    assert _spec(native).description == "Echo input"


def test_mcp_tool_to_native_with_prefix() -> None:
    """Tool prefix is prepended to the name."""
    mcp_tool = _fake_mcp_tool(name="echo")
    native = _mcp_tool_to_native(
        mcp_tool, "test:key", _noop_transport_factory, "ctx7"
    )

    assert native.name == "ctx7_echo"


def test_mcp_tool_to_native_schema_preserved() -> None:
    """The inputSchema from the MCP tool is passed through as params."""
    mcp_tool = _fake_mcp_tool()
    native = _mcp_tool_to_native(
        mcp_tool, "test:key", _noop_transport_factory, None
    )

    assert _spec(native).params == mcp_tool.input_schema
    assert _spec(native).description == "Echo input"


async def test_get_http_tools_raises_installation_error_without_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "mcp.client.streamable_http":
            raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ai.InstallationError) as exc_info:
        await ai.mcp.get_http_tools("https://mcp.example.com/mcp")

    assert "ai[mcp]" in str(exc_info.value)


# -- End-to-end: MCP tool executes through Agent default loop ---------------


async def test_mcp_tool_executes_through_agent() -> None:
    """MCP-style tool via _mcp_tool_to_native works with Agent."""
    call_log: list[dict[str, str]] = []

    async def fake_fn(**kwargs: str) -> str:
        call_log.append(kwargs)
        return f"echoed: {kwargs.get('text', '')}"

    # Build a tool the same way the MCP client does,
    # but with a fake fn so we don't need a real MCP server.
    mcp_tool = _fake_mcp_tool(name="mcp_e2e_echo")
    native = _mcp_tool_to_native(
        mcp_tool, "test:key", _noop_transport_factory, None
    )
    # Replace the executor (which would try to connect) with our fake.
    native = dataclasses.replace(native, fn=fake_fn)

    my_agent = ai.Agent(tools=[native])

    call1 = [
        tool_call_msg(
            tc_id="tc-mcp-1", name="mcp_e2e_echo", args='{"text": "hello"}'
        )
    ]
    call2 = [text_msg("Done.", id="msg-2")]
    llm = mock_llm([call1, call2])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("echo hello")]
    ) as stream:
        msgs = await collect_messages(stream)

    # Tool was called with the right args.
    assert len(call_log) == 1
    assert call_log[0] == {"text": "hello"}

    # Tool result is visible in messages.
    tool_results = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_results) >= 1
    tr = tool_results[0].tool_results[0].result
    assert tr == "echoed: hello"

    # LLM was called twice (tool call + final text).
    assert llm.call_count == 2


def _spec(tool: ai.AgentTool) -> ai.tools.ToolSpec:
    spec = tool.tool.spec
    assert spec is not None
    return spec


async def test_mcp_connection_created_lazily_by_tool_call() -> None:
    """A connection opened from inside a tool call (whose transport runs
    a task group) survives the tool call and is closed at the end of
    the run."""
    server = MCPServer("test")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    connects = 0

    def transport_factory() -> contextlib.AbstractAsyncContextManager[Any]:
        nonlocal connects
        connects += 1
        return InMemoryTransport(server)

    mcp_tool = mcp.types.Tool(
        name="add",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    )
    native = _mcp_tool_to_native(mcp_tool, "mem:add", transport_factory, None)

    mock_llm(
        [
            [tool_call_msg(tc_id="tc-1", name="add", args='{"a": 1, "b": 2}')],
            [tool_call_msg(tc_id="tc-2", name="add", args='{"a": 3, "b": 4}')],
            [text_msg("Done.", id="msg-3")],
        ]
    )
    async with ai.Agent(tools=[native]).run(
        MOCK_MODEL, [ai.user_message("add")]
    ) as stream:
        msgs = await collect_messages(stream)

    results = [r.result for m in msgs for r in m.tool_results]
    assert results == [{"result": 3}, {"result": 7}]
    assert connects == 1
