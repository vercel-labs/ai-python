"""Custom loop: manual control over streaming and tool execution."""

import asyncio
from collections.abc import AsyncGenerator

import ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    await asyncio.sleep(2)
    return f"Sunny, 72F in {city}" if city == "Tokyo" else f"Cloudy, 55F in {city}"


@ai.tool
async def get_population(city: str) -> int:
    """Get population of a city."""
    await asyncio.sleep(1)
    return {"new york": 8_336_817, "tokyo": 13_960_000}.get(city.lower(), 1_000_000)


class CustomAgent(ai.Agent):
    TOOLS = [get_weather, get_population]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        """Stream, execute tools with logging, repeat."""
        while context.keep_running():
            async with (
                ai.stream(context=context) as stream,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(stream, tr.events()):
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        call = event.tool_call
                        print(f"Launching tool {call.tool_name}({call.tool_args})")
                        tool = context.resolve(call)
                        tr.schedule(tool)

                context.add(stream.message)
                # This adds the tool message to the history, which
                # also has the effect of causing another turn through
                # the loop.
                context.add(tr.get_tool_message())


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    my_agent = CustomAgent()

    async with my_agent.run(
        model,
        [ai.user_message("Compare the weather and population of New York and Tokyo.")],
    ) as stream:
        async for event in stream:
            if (
                isinstance(event, ai.events.StreamEnd)
                and event.message.role == "assistant"
            ):
                print("====", event.message.text, flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
