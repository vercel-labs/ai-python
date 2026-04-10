"""MCP client: tool conversion, end-to-end execution."""

from __future__ import annotations

import contextlib
from typing import Any

import mcp.types

import ai
from ai.agents.mcp.client import _mcp_tool_to_native

from ...conftest import MOCK_MODEL, mock_llm, text_msg, tool_call_msg


def _fake_mcp_tool(
    name: str = "mcp_echo", description: str = "Echo input"
) -> mcp.types.Tool:
    """Build a minimal mcp.types.Tool for testing."""
    return mcp.types.Tool(
        name=name,
        description=description,
        inputSchema={
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
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, None)

    assert native.name == "mcp_basic_test"
    assert native.description == "Echo input"


def test_mcp_tool_to_native_with_prefix() -> None:
    """Tool prefix is prepended to the name."""
    mcp_tool = _fake_mcp_tool(name="echo")
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, "ctx7")

    assert native.name == "ctx7_echo"


def test_mcp_tool_to_native_schema_preserved() -> None:
    """The inputSchema from the MCP tool is passed through as param_schema."""
    mcp_tool = _fake_mcp_tool()
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, None)

    assert native.param_schema == mcp_tool.inputSchema
    assert native.description == "Echo input"


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
    native = _mcp_tool_to_native(mcp_tool, "test:key", _noop_transport_factory, None)
    # Replace the real fn (which would try to connect) with our fake.
    native._fn = fake_fn

    my_agent = ai.agent(tools=[native])

    call1 = [
        tool_call_msg(tc_id="tc-mcp-1", name="mcp_e2e_echo", args='{"text": "hello"}')
    ]
    call2 = [text_msg("Done.", id="msg-2")]
    llm = mock_llm([call1, call2])

    msgs: list[ai.Message] = []
    async for msg in my_agent.run(MOCK_MODEL, [ai.user_message("echo hello")]):
        msgs.append(msg)

    # Tool was called with the right args.
    assert len(call_log) == 1
    assert call_log[0] == {"text": "hello"}

    # Tool result is visible in messages.
    tool_results = [m for m in msgs if m.role == "tool" and m.tool_results]
    assert len(tool_results) >= 1
    assert tool_results[0].tool_results[0].result == "echoed: hello"

    # LLM was called twice (tool call + final text).
    assert llm.call_count == 2
