"""Middleware: run-scoped registration, chain ordering, per-surface wrapping."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncGenerator
from typing import Any

import pydantic
import pytest

import ai
from ai.agents import _middleware as middleware
from ai.types import events as agent_events_

from .conftest import (
    MOCK_MODEL,
    collect_messages,
    mock_llm,
    text_msg,
    tool_call_msg,
)

# ── Helpers ──────────────────────────────────────────────────────


class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


# ── wrap_tool ───────────────────────────────────────────────────


async def test_wrap_tool_is_called() -> None:
    """Middleware.wrap_tool is invoked for every tool execution."""
    tool_calls: list[middleware.ToolContext] = []

    class Spy(middleware._Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            tool_calls.append(call)
            return await next(call)

    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    my_agent = ai.agent(tools=[double])
    call1 = [tool_call_msg(tc_id="tc-1", name="double", args='{"x": 7}')]
    call2 = [text_msg("14")]
    mock_llm([call1, call2])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("Double 7")], _middleware=[Spy()]
    ) as stream:
        async for _m in stream:
            pass

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "double"
    assert tool_calls[0].tool_call_id == "tc-1"
    assert tool_calls[0].kwargs == {"x": 7}


# ── wrap_hook ───────────────────────────────────────────────────


async def test_wrap_hook_is_called() -> None:
    """Middleware.wrap_hook is invoked for every ai.hook() call."""
    hook_calls: list[middleware.HookContext] = []

    class Spy(middleware._Middleware):
        async def wrap_hook(self, call: middleware.HookContext, next: Any) -> Any:
            hook_calls.append(call)
            return await next(call)

    class MyAgent(ai.Agent):
        async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.Event]:
            async with ai.models.stream(context=context) as stream:
                async for event in stream:
                    yield event
            await ai.hook("test_hook", payload=Confirmation)

    my_agent = MyAgent()

    mock_llm([[text_msg("OK")]])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], _middleware=[Spy()]
    ) as stream:
        async for event in stream:
            if not isinstance(event, agent_events_.HookEvent):
                continue
            if event.hook.status == "pending":
                ai.resolve_hook("test_hook", {"approved": True, "reason": "ok"})

    assert len(hook_calls) == 1
    assert hook_calls[0].label == "test_hook"
    assert hook_calls[0].payload is Confirmation


# ── wrap_agent_run ──────────────────────────────────────────────


async def test_wrap_agent_run_ordering() -> None:
    """Agent run middleware chain runs in the correct onion order."""
    order: list[str] = []

    class Outer(middleware._Middleware):
        async def wrap_agent_run(
            self, call: ai.Context, next: Any
        ) -> AsyncGenerator[ai.events.Event]:
            order.append("outer-before")
            async for event in next(call):
                yield event
            order.append("outer-after")

    class Inner(middleware._Middleware):
        async def wrap_agent_run(
            self, call: ai.Context, next: Any
        ) -> AsyncGenerator[ai.events.Event]:
            order.append("inner-before")
            async for event in next(call):
                yield event
            order.append("inner-after")

    my_agent = ai.agent()
    mock_llm([[text_msg("Hi")]])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("Hi")], _middleware=[Outer(), Inner()]
    ) as stream:
        async for _m in stream:
            pass

    assert order == ["outer-before", "inner-before", "inner-after", "outer-after"]


async def test_wrap_tool_context_fields_flow_to_result() -> None:
    """ToolContext.tool_name is used in the result message."""

    class Rewriter(middleware._Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            # Rewrite the tool_name via dataclasses.replace.
            modified = dataclasses.replace(call, tool_name="rewritten-name")
            return await next(modified)

    @ai.tool
    async def echo(x: int) -> int:
        """Echo a number."""
        return x

    my_agent = ai.agent(tools=[echo])
    call1 = [tool_call_msg(tc_id="tc-1", name="echo", args='{"x": 42}')]
    call2 = [text_msg("done")]
    mock_llm([call1, call2])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], _middleware=[Rewriter()]
    ) as stream:
        msgs = await collect_messages(stream)
    tool_result_msgs = [m for m in msgs if m.role == "tool" and m.tool_results]

    assert len(tool_result_msgs) >= 1
    # The result message should use the rewritten name, not the original.
    assert tool_result_msgs[0].tool_results[0].tool_name == "rewritten-name"


async def test_wrap_tool_rewriting_tool_call_id_breaks_history() -> None:
    """tool_call_id is a correlation key and must stay stable."""

    class Rewriter(middleware._Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            modified = dataclasses.replace(call, tool_call_id="rewritten-id")
            return await next(modified)

    @ai.tool
    async def echo(x: int) -> int:
        """Echo a number."""
        return x

    my_agent = ai.agent(tools=[echo])
    call1 = [tool_call_msg(tc_id="original-id", name="echo", args='{"x": 42}')]
    call2 = [text_msg("done")]
    mock_llm([call1, call2])

    with pytest.raises(ExceptionGroup) as exc_info:
        async with my_agent.run(
            MOCK_MODEL, [ai.user_message("go")], _middleware=[Rewriter()]
        ) as stream:
            async for _m in stream:
                pass

    assert len(exc_info.value.exceptions) == 1
    assert "orphaned-tool-result" in str(exc_info.value.exceptions[0])


# ── Context snapshot isolation ──────────────────────────────────


async def test_model_context_messages_are_isolated() -> None:
    """Mutating call.messages in middleware does not affect the caller."""
    original_messages = [ai.user_message("Hello")]

    class Mutator(middleware._Middleware):
        async def wrap_model(self, call: middleware.ModelContext, next: Any) -> Any:
            # Try to mutate the context's messages list in place.
            call.messages.append(ai.system_message("injected"))
            return await next(call)

    my_agent = ai.agent()
    mock_llm([[text_msg("Hi")]])

    async with my_agent.run(
        MOCK_MODEL, original_messages, _middleware=[Mutator()]
    ) as stream:
        async for _m in stream:
            pass

    # The original list should be unmodified.
    assert len(original_messages) == 1
    assert original_messages[0].role == "user"


# ── Middleware can repair bad tool args ──────────────────────────


async def test_middleware_can_fix_bad_tool_kwargs() -> None:
    """A middleware that rewrites call.kwargs can fix malformed tool args."""

    class ArgFixer(middleware._Middleware):
        async def wrap_tool(self, call: middleware.ToolContext, next: Any) -> Any:
            # If kwargs are empty (parse failed), supply valid ones.
            if not call.kwargs:
                fixed = dataclasses.replace(call, kwargs={"x": 99})
                return await next(fixed)
            return await next(call)

    @ai.tool
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    my_agent = ai.agent(tools=[double])
    # Send completely invalid JSON args — parse will fail, kwargs will be {}.
    call1 = [tool_call_msg(tc_id="tc-bad", name="double", args="not json")]
    call2 = [text_msg("done")]
    mock_llm([call1, call2])

    async with my_agent.run(
        MOCK_MODEL, [ai.user_message("go")], _middleware=[ArgFixer()]
    ) as stream:
        msgs = await collect_messages(stream)
    tool_result_msgs = [m for m in msgs if m.role == "tool" and m.tool_results]

    assert len(tool_result_msgs) >= 1
    # The fixer middleware supplied x=99, so double should return 198.
    assert tool_result_msgs[0].tool_results[0].result == 198
    assert tool_result_msgs[0].tool_results[0].is_error is False
