"""Simplest agent: model + tool, default loop handles everything."""

import asyncio

import ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


async def main() -> None:
    model = ai.get_model("anthropic/claude-sonnet-4.6")

    my_agent = ai.Agent(tools=[get_weather])

    messages = [
        ai.system_message("You are a helpful weather assistant."),
        ai.user_message("What's the weather in Tokyo?"),
    ]

    async with my_agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            if isinstance(event, ai.events.StreamEnd):
                print()
    print()


if __name__ == "__main__":
    asyncio.run(main())
