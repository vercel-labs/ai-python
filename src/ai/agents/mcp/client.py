from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    import mcp.client.session
    import mcp.types

from ... import types
from ..agent import AgentTool, Tool

__all__ = [
    "close_connections",
    "get_http_tools",
    "get_stdio_tools",
]


@dataclasses.dataclass
class _Connection:
    """Internal connection state - never exposed to users."""

    client: mcp.client.session.ClientSession
    exit_stack: contextlib.AsyncExitStack


# Connection pool stored in contextvar, scoped to Agent.run()
_pool: contextvars.ContextVar[dict[str, _Connection] | None] = (
    contextvars.ContextVar("mcp_connections", default=None)
)

_pool_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def ensure_connection_pool() -> AsyncIterator[dict[str, _Connection]]:
    pool = orig_pool = _pool.get()
    if pool is None:
        pool = {}
        _pool.set(pool)
    try:
        yield pool
    finally:
        if orig_pool is None:
            await close_connections()
        _pool.set(orig_pool)


async def _get_or_create_connection(
    key: str,
    transport_factory: Callable[
        [], contextlib.AbstractAsyncContextManager[Any]
    ],
) -> mcp.client.session.ClientSession:
    """Get an existing connection or create a new one."""
    import mcp.client.session as _mcp_session  # noqa: PLC0415

    pool = _pool.get()

    if pool is None:
        raise RuntimeError(
            "MCP tools must be used inside agent.run(). "
            "The connection pool is not initialized."
        )

    async with _pool_lock:
        if key in pool:
            return pool[key].client

        exit_stack = contextlib.AsyncExitStack()

        try:
            streams = await exit_stack.enter_async_context(transport_factory())
            read_stream, write_stream = streams[0], streams[1]

            client = _mcp_session.ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
            )
            await exit_stack.enter_async_context(client)
            await client.initialize()

            pool[key] = _Connection(client=client, exit_stack=exit_stack)
            return client

        except BaseException:
            await exit_stack.aclose()
            raise


def _make_tool_fn(
    connection_key: str,
    tool_name: str,
    transport_factory: Callable[
        [], contextlib.AbstractAsyncContextManager[Any]
    ],
) -> Callable[..., Awaitable[Any]]:
    """Create a tool function that manages its own connection."""

    async def call_tool(**kwargs: Any) -> Any:
        import mcp.types as _mcp_types  # noqa: PLC0415

        client = await _get_or_create_connection(
            connection_key, transport_factory
        )
        try:
            result = await asyncio.wait_for(
                client.call_tool(tool_name, kwargs),
                timeout=30.0,
            )
        except TimeoutError as e:
            raise RuntimeError(
                f"MCP tool call timed out after 30 seconds: {tool_name}"
            ) from e

        if result.isError:
            error_text = " ".join(
                part.text
                for part in result.content
                if isinstance(part, _mcp_types.TextContent)
            )
            raise RuntimeError(
                f"MCP tool error: {error_text or 'Unknown error'}"
            )

        if result.structuredContent is not None:
            return result.structuredContent

        for part in result.content:
            if isinstance(part, _mcp_types.TextContent):
                text = part.text
                if text.startswith(("{", "[")):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
                return text

        return result.content

    return call_tool


async def get_stdio_tools(
    command: str,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    tool_prefix: str | None = None,
) -> list[AgentTool]:
    """Get tools from an MCP server running as a subprocess.

    Connection is managed automatically - created on first use, cleaned up
    when the agent run finishes.

    Args:
        command: The command to run (e.g., "npx", "python").
        *args: Arguments to pass to the command.
        env: Environment variables for the subprocess.
        cwd: Working directory for the subprocess.
        tool_prefix: Optional prefix to add to all tool names.

    Returns:
        List of AgentTool objects that can be passed to an agent.

    Example::

        tools = await ai.mcp.get_stdio_tools(
            "npx", "-y", "@anthropic/mcp-server-filesystem", "/tmp"
        )

    """
    import mcp.client.stdio as _mcp_stdio  # noqa: PLC0415

    connection_key = f"stdio:{command}:{':'.join(args)}"

    def transport_factory() -> contextlib.AbstractAsyncContextManager[Any]:
        return _mcp_stdio.stdio_client(
            _mcp_stdio.StdioServerParameters(
                command=command,
                args=list(args),
                env=env,
                cwd=cwd,
            )
        )

    client = await _get_or_create_connection(connection_key, transport_factory)
    result = await client.list_tools()

    return [
        _mcp_tool_to_native(
            mcp_tool, connection_key, transport_factory, tool_prefix
        )
        for mcp_tool in result.tools
    ]


async def get_http_tools(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    tool_prefix: str | None = None,
) -> list[AgentTool]:
    """Get tools from an MCP server over HTTP (Streamable HTTP transport).

    Connection is managed automatically - created on first use, cleaned up
    when the agent run finishes.

    Args:
        url: The URL of the MCP server endpoint.
        headers: Optional HTTP headers (e.g., for authentication).
        tool_prefix: Optional prefix to add to all tool names.

    Returns:
        List of AgentTool objects that can be passed to an agent.

    Example::

        tools = await ai.mcp.get_http_tools(
            "http://localhost:3000/mcp",
            headers={"Authorization": "Bearer xxx"}
        )

    """
    import httpx as _httpx  # noqa: PLC0415
    import mcp.client.streamable_http as _mcp_http  # noqa: PLC0415

    connection_key = f"http:{url}"

    def transport_factory() -> contextlib.AbstractAsyncContextManager[Any]:
        http_client = _httpx.AsyncClient(headers=headers) if headers else None
        return _mcp_http.streamable_http_client(
            url=url, http_client=http_client
        )

    async with ensure_connection_pool():
        client = await _get_or_create_connection(
            connection_key, transport_factory
        )
        result = await client.list_tools()

    return [
        _mcp_tool_to_native(
            mcp_tool, connection_key, transport_factory, tool_prefix
        )
        for mcp_tool in result.tools
    ]


def _mcp_tool_to_native(
    mcp_tool: mcp.types.Tool,
    connection_key: str,
    transport_factory: Callable[
        [], contextlib.AbstractAsyncContextManager[Any]
    ],
    tool_prefix: str | None,
) -> AgentTool:
    """Convert an MCP tool to a native AgentTool.

    ``mcp_tool`` is typed as :class:`mcp.types.Tool` for static analysis;
    the actual ``mcp.types`` import is deferred to the calling function.
    """
    name = mcp_tool.name
    if tool_prefix:
        name = f"{tool_prefix}_{name}"

    tool = Tool(
        kind="function",
        name=name,
        args=types.tools.FunctionToolArgs(
            description=mcp_tool.description or "",
            params=mcp_tool.inputSchema,
        ),
    )

    return AgentTool(
        tool=tool,
        fn=_make_tool_fn(connection_key, mcp_tool.name, transport_factory),
    )


async def close_connections() -> None:
    """Close all MCP connections in the current context.

    Called automatically at the end of an agent run, but can also be
    called manually for explicit cleanup.
    """
    pool = _pool.get()
    if pool is None:
        return

    async with _pool_lock:
        if not pool:
            return

        for conn in pool.values():
            await _close_connection_safely(conn)

        pool.clear()


async def _close_connection_safely(conn: _Connection) -> None:
    """Close a connection, suppressing any errors."""
    with contextlib.suppress(Exception):
        await conn.exit_stack.aclose()
