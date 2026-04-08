"""Custom agent loop with @ai.stream and manual tool execution."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import vercel_ai_sdk as ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    return {"new york": 8_336_817, "los angeles": 3_979_576}.get(
        city.lower(), 1_000_000
    )


@ai.stream
async def custom_stream_step(
    model: ai.Model,
    messages: list[ai.Message],
    tools: list[ai.Tool[..., Any]],
    label: str | None = None,
) -> AsyncGenerator[ai.Message]:
    """Wraps models.stream to inject a label on every message."""
    async for msg in await ai.models.stream(model, messages, tools=tools):
        msg.label = label
        yield msg


async def main() -> None:
    model = ai.Model(
        id="anthropic/claude-opus-4.6",
        adapter="ai-gateway-v3",
        provider="ai-gateway",
    )

    my_agent = ai.agent(
        model=model,
        system="Answer questions using the weather and population tools.",
        tools=[get_weather, get_population],
    )

    @my_agent.loop
    async def custom(agent: ai.Agent, messages: list[ai.Message]) -> ai.StreamResult:
        """Custom agent loop with manual tool execution.

        Uses @ai.stream for custom streaming and
        asyncio.gather for parallel tool execution.
        """
        local_messages = list(messages)

        while True:
            result = await custom_stream_step(
                agent.model, local_messages, agent.tools, label="agent"
            )

            if not result.tool_calls:
                return result

            if result.last_message is not None:
                local_messages.append(result.last_message)
            await asyncio.gather(
                *(
                    ai.execute_tool(tc, message=result.last_message)
                    for tc in result.tool_calls
                )
            )

    async for msg in my_agent.run(
        [
            ai.user_message(
                "What's the weather and population of New York and Los Angeles?"
            )
        ]
    ):
        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
