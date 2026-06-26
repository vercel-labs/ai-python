"""Streaming from inside a tool via an async generator.

An async generator tool yields values that flow through the runtime
sink to the consumer in real time as ``PartialToolCallResult`` events.
The tool's aggregator decides how the yielded values are combined into
the final tool result that goes back to the model.

Here the aggregator is declared via the :data:`ai.StreamingStatusTool`
return-type alias. This is equivalent to declaring the aggregator in
``@ai.tool(...)``.
"""

import asyncio

import ai


@ai.tool
async def talk_to_mothership(question: str) -> ai.StreamingStatusTool[str]:
    """Ask the mothership a question. Streams progress back to the caller."""
    for step in [
        "Connecting...",
        "Transmitting...",
        f"Asking: {question!r}...",
    ]:
        yield step
        await asyncio.sleep(0.3)

    # LastAggregator keeps only the final yield, which becomes the tool result.
    yield "The mothership says: Soon."


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")

    my_agent = ai.agent(tools=[talk_to_mothership])

    messages = [
        ai.system_message(
            "Use the mothership tool when asked about the future."
        ),
        ai.user_message("When will the robots take over?"),
    ]

    async with my_agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.PartialToolCallResult):
                print(f"  [{event.value}]")
            elif isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
