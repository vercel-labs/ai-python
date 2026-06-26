"""Nested agents with a research tool that streams through the sink."""

import asyncio

import ai

model = ai.get_model("anthropic/claude-sonnet-4.6")


@ai.tool
async def get_facts(topic: str) -> str:
    """Look up facts about a topic."""
    facts = {
        "mars": (
            "Mars has two moons: Phobos and Deimos. "
            "A day on Mars is 24.6 hours."
        ),
        "venus": "Venus rotates backwards. Its surface temperature is 450C.",
    }
    return facts.get(topic.lower(), f"No facts found for {topic}.")


# The SubAgentTool alias declares MessageAggregator on the return type:
# events stream through the runtime sink, the collected messages flow to
# the consumer, and the final assistant text becomes the tool result.
@ai.tool
async def research(topic: str) -> ai.SubAgentTool:
    """Research a topic in depth using a sub-agent."""
    researcher = ai.agent(tools=[get_facts])

    messages = [
        ai.system_message("You are a research assistant. Be concise."),
        ai.user_message(f"Research: {topic}"),
    ]

    async with researcher.run(model, messages) as stream:
        async for event in stream:
            yield event


async def main() -> None:
    orchestrator = ai.agent(tools=[research])

    messages = [
        ai.system_message(
            "Use the research tool to answer questions. Summarize the findings."
        ),
        ai.user_message("Tell me about Mars."),
    ]

    async with orchestrator.run(model, messages) as stream:
        async for event in stream:
            # Subtool results
            if isinstance(event, ai.events.PartialToolCallResult):
                if isinstance(event.value, ai.events.TextDelta):
                    print(event.value.chunk.upper(), end="", flush=True)
                elif isinstance(event.value, ai.events.StreamEnd):
                    print()
            elif isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
