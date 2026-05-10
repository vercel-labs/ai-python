"""Middleware example: a print-based middleware that logs all execution surfaces.

Demonstrates all five middleware surfaces:
- wrap_agent_run  — the entire agent run
- wrap_model      — each LLM streaming call
- wrap_generate   — non-streaming generation (images, video, etc.)
- wrap_tool       — each tool execution
- wrap_hook       — each hook suspension point

Middleware is passed to agent.run() and applies to all execution surfaces
within that run — including nested model calls, tool calls, and hooks.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import pydantic

import ai


class PrintMiddleware(ai.Middleware):
    """Logs every execution surface to stdout."""

    async def wrap_agent_run(
        self,
        call: ai.middleware.AgentRunContext,
        next: Callable[[ai.middleware.AgentRunContext], AsyncGenerator[Any]],
    ) -> AsyncGenerator[Any]:
        print(f">>> [run] agent starting  tools={len(call.tools)}")
        t0 = time.perf_counter()

        async for event in next(call):
            yield event

        elapsed = time.perf_counter() - t0
        print(f"<<< [run] agent finished  {elapsed:.2f}s")

    async def wrap_model(
        self,
        call: ai.middleware.ModelContext,
        next: Callable[[ai.middleware.ModelContext], Awaitable[Any]],
    ) -> Any:
        print(f"\n>>> [model] calling {call.model.id}")
        print(f"    messages: {len(call.messages)}")
        if call.tools:
            print(f"    tools: {[t.name for t in call.tools]}")

        result = await next(call)

        print("<<< [model] stream started")
        return result

    async def wrap_generate(
        self,
        call: ai.middleware.GenerateContext,
        next: Callable[[ai.middleware.GenerateContext], Awaitable[ai.messages.Message]],
    ) -> ai.messages.Message:
        print(f"\n>>> [generate] calling {call.model.id}")
        print(f"    messages: {len(call.messages)}")

        result = await next(call)

        print("<<< [generate] done")
        return result

    async def wrap_tool(
        self,
        call: ai.middleware.ToolContext,
        next: Callable[
            [ai.middleware.ToolContext], Awaitable[ai.events.ToolCallResult]
        ],
    ) -> ai.events.ToolCallResult:
        print(f"\n>>> [tool] {call.tool_name}({call.kwargs})")

        result = await next(call)

        tr = result.results[0] if result.results else None
        if tr and not tr.is_error:
            print(f"<<< [tool] {call.tool_name} -> {tr.result}")
        elif tr:
            print(f"<<< [tool] {call.tool_name} ERROR: {tr.result}")
        return result

    async def wrap_hook(
        self,
        call: ai.middleware.HookContext,
        next: Callable[[ai.middleware.HookContext], Awaitable[pydantic.BaseModel]],
    ) -> pydantic.BaseModel:
        print(f"\n>>> [hook] {call.label}  payload={call.payload.__name__}")

        result = await next(call)

        print(f"<<< [hook] {call.label} resolved")
        return result


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    return {"tokyo": 13_960_000, "new york": 8_336_817}.get(city.lower(), 1_000_000)


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[get_weather, get_population])

    messages = [
        ai.system_message("You are a helpful assistant. Use tools when needed."),
        ai.user_message("What's the weather and population of Tokyo?"),
    ]

    print("--- starting agent run ---\n")
    async with my_agent.run(model, messages, middleware=[PrintMiddleware()]) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print("\n\n--- done ---")


if __name__ == "__main__":
    asyncio.run(main())
