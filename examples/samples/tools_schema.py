"""Tools — pass tool schemas to the model."""

import asyncio

import ai

model = ai.get_model("gateway:anthropic/claude-sonnet-4")

# Define a schema-only tool.
get_weather = ai.Tool(
    kind="function",
    name="get_weather",
    args=ai.tools.FunctionToolArgs(
        description="Get the current weather for a city.",
        params={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    ),
)

messages = [ai.user_message("What's the weather in Tokyo?")]


async def main() -> None:
    # Stream with tools — the model may emit tool calls.
    async with ai.stream(model, messages, tools=[get_weather]) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()

    # After iteration, s.tool_calls collects every tool call from the response.
    for tc in s.tool_calls:
        print(f"Tool call: {tc.tool_name}({tc.tool_args})")


if __name__ == "__main__":
    asyncio.run(main())
