"""Nested agents: a research tool that is itself an agent streaming through the sink."""

import asyncio
from collections.abc import AsyncGenerator

import ai

model = ai.ai_gateway("anthropic/claude-sonnet-4")


@ai.tool
async def get_facts(topic: str) -> str:
    """Look up facts about a topic."""
    facts = {
        "mars": "Mars has two moons: Phobos and Deimos. A day on Mars is 24.6 hours.",
        "venus": "Venus rotates backwards. Its surface temperature is 450C.",
    }
    return facts.get(topic.lower(), f"No facts found for {topic}.")


# This tool is an async generator — it streams intermediate messages
# through the runtime sink, then returns the final result.
@ai.tool  # type: ignore[arg-type]  # async generator tools are supported at runtime
async def research(topic: str) -> AsyncGenerator[ai.AgentEvent]:
    """Research a topic in depth using a sub-agent."""
    researcher = ai.agent(tools=[get_facts])

    messages = [
        ai.system_message("You are a research assistant. Be concise."),
        ai.user_message(f"Research: {topic}"),
    ]

    async for event in researcher.run(model, messages):
        yield event


async def main() -> None:
    orchestrator = ai.agent(tools=[research])

    messages = [
        ai.system_message(
            "Use the research tool to answer questions. Summarize the findings."
        ),
        ai.user_message("Tell me about Mars."),
    ]

    async for event in orchestrator.run(model, messages):
        # Subtool results
        if isinstance(event, ai.PartialToolCallResult):
            if isinstance(event.value, ai.TextDelta):
                print(event.value.chunk.upper(), end="", flush=True)
            elif isinstance(event, ai.StreamEnd):
                print()

        if isinstance(event, ai.TextDelta):
            print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
