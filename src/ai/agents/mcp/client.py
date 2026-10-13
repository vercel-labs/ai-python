from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import importlib
import json
from typing import TYPE_CHECKING, Any, cast

import anyio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    import mcp.client.session
    import mcp.types

from ... import errors, types, util
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
    close: anyio.Event
    closed: anyio.Event


@dataclasses.dataclass
class _Pool:
    # Each connection is served by a task in this group, since the
    # transports hold task groups that must be entered and exited in
    # the same task.
    tg: anyio.abc.TaskGroup
    connections: dict[str, _Connection] = dataclasses.field(
        default_factory=dict
    )


# Connection pool stored in contextvar, scoped to Agent.run()
_pool: contextvars.ContextVar[_Pool | None] = contextvars.ContextVar(
    "mcp_connections", default=None
)

_pool_lock = anyio.Lock()


def _import_mcp_module(module_name: str) -> Any:
    """Import an MCP module or raise the public optional dependency error."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        root_module = module_name.partition(".")[0]
        if exc.name not in {module_name, root_module}:
            raise
        raise errors.InstallationError(
            "could not import `mcp`, which is required to use MCP tools, "
            'you can install it with `pip install "ai[mcp]"` or '
            '`uv add "ai[mcp]"`'
        ) from exc


@contextlib.asynccontextmanager
async def connection_pool() -> AsyncIterator[_Pool]:
    """Open a fresh connection pool, closing its connections on exit."""
    async with util.create_task_group() as tg:
        token = _pool.set(pool := _Pool(tg))
        try:
            yield pool
        finally:
            await close_connections()
            _pool.reset(token)


@contextlib.asynccontextmanager
async def ensure_connection_pool() -> AsyncIterator[_Pool]:
    if (pool := _pool.get()) is not None:
        yield pool
    else:
        async with connection_pool() as pool:
            yield pool


async def _serve_connection(
    transport_factory: Callable[
        [], contextlib.AbstractAsyncContextManager[Any]
    ],
    *,
    task_status: anyio.abc.TaskStatus[_Connection],
) -> None:
    mcp_session = _import_mcp_module("mcp.client.session")
    client_session = cast(
        "type[mcp.client.session.ClientSession]", mcp_session.ClientSession
    )

    conn: _Connection | None = None
    closed = anyio.Event()
    try:
        async with transport_factory() as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with client_session(
                read_stream=read_stream,
                write_stream=write_stream,
            ) as client:
                await client.initialize()
                conn = _Connection(
                    client=client, close=anyio.Event(), closed=closed
                )
                task_status.started(conn)
                await conn.close.wait()
    except Exception:
        # Errors while closing an established connection are ignored.
        if conn is None:
            raise
    finally:
        closed.set()


async def _get_or_create_connection(
    key: str,
    transport_factory: Callable[
        [], contextlib.AbstractAsyncContextManager[Any]
    ],
) -> mcp.client.session.ClientSession:
    """Get an existing connection or create a new one."""
    pool = _pool.get()

    if pool is None:
        raise RuntimeError(
            "MCP tools must be used inside agent.run(). "
            "The connection pool is not initialized."
        )

    async with _pool_lock:
        if key in pool.connections:
            return pool.connections[key].client

        conn: _Connection = await pool.tg.start(
            _serve_connection, transport_factory
        )
        pool.connections[key] = conn
        return conn.client


def _make_tool_fn(
    connection_key: str,
    tool_name: str,
    transport_factory: Callable[
        [], contextlib.AbstractAsyncContextManager[Any]
    ],
) -> Callable[..., Awaitable[Any]]:
    """Create a tool function that manages its own connection."""

    async def call_tool(**kwargs: Any) -> Any:
        mcp_types = _import_mcp_module("mcp.types")
        text_content = cast(
            "type[mcp.types.TextContent]", mcp_types.TextContent
        )

        client = await _get_or_create_connection(
            connection_key, transport_factory
        )
        try:
            with anyio.fail_after(30.0):
                result = await client.call_tool(tool_name, kwargs)
        except TimeoutError as e:
            raise RuntimeError(
                f"MCP tool call timed out after 30 seconds: {tool_name}"
            ) from e

        if result.is_error:
            error_text = " ".join(
                part.text
                for part in result.content
                if isinstance(part, text_content)
            )
            raise RuntimeError(
                f"MCP tool error: {error_text or 'Unknown error'}"
            )

        if result.structured_content is not None:
            return result.structured_content

        for part in result.content:
            if isinstance(part, text_content):
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
    mcp_stdio = _import_mcp_module("mcp.client.stdio")

    connection_key = f"stdio:{command}:{':'.join(args)}"

    def transport_factory() -> contextlib.AbstractAsyncContextManager[Any]:
        return cast(
            "contextlib.AbstractAsyncContextManager[Any]",
            mcp_stdio.stdio_client(
                mcp_stdio.StdioServerParameters(
                    command=command,
                    args=list(args),
                    env=env,
                    cwd=cwd,
                )
            ),
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
    import httpx2 as httpx  # noqa: PLC0415

    mcp_http = _import_mcp_module("mcp.client.streamable_http")

    connection_key = f"http:{url}"

    def transport_factory() -> contextlib.AbstractAsyncContextManager[Any]:
        http_client = httpx.AsyncClient(headers=headers) if headers else None
        return cast(
            "contextlib.AbstractAsyncContextManager[Any]",
            mcp_http.streamable_http_client(url=url, http_client=http_client),
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
        spec=types.tools.ToolSpec(
            description=mcp_tool.description or "",
            params=mcp_tool.input_schema,
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
        conns = list(pool.connections.values())
        pool.connections.clear()
        for conn in conns:
            conn.close.set()
        for conn in conns:
            await conn.closed.wait()
