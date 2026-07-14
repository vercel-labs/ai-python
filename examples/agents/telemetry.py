"""Agent run with telemetry: console adapter + a custom span in a tool.

For an out-of-process viewer, run
``python -m ai.experimental_telemetry.utils.viewer`` and see
``ai/experimental_telemetry/utils/viewer.py`` for the otel exporter
setup.
"""

import asyncio

import ai
from ai.experimental_telemetry.utils import console


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    async with ai.experimental_telemetry.span("lookup", city=city) as span:
        await asyncio.sleep(0.1)
        span.set(source="cache")
    return f"Sunny, 72F in {city}"


async def main() -> None:
    ai.experimental_telemetry.register(console.ConsoleAdapter())

    model = ai.get_model("anthropic/claude-sonnet-4.6")
    my_agent = ai.Agent(tools=[get_weather])
    messages = [ai.user_message("What's the weather in Tokyo?")]

    async with my_agent.run(model, messages) as stream:
        async for _ in stream:
            pass


if __name__ == "__main__":
    asyncio.run(main())
