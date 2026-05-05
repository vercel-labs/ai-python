"""Streaming from inside a tool via an async generator.

An async generator tool yields events that flow through the runtime
sink to the consumer in real time. The final yielded message's text
becomes the tool result.
"""

import asyncio
from collections.abc import AsyncGenerator

import ai


@ai.tool  # type: ignore[arg-type]  # async generator tools are supported at runtime
async def talk_to_mothership(
    question: str,
) -> AsyncGenerator[ai.AgentEvent | ai.Message]:
    """Ask the mothership a question. Streams progress back to the caller."""
    for step in ["Connecting...", "Transmitting...", "Awaiting response..."]:
        yield ai.TextStart(block_id=f"progress-{step}")
        yield ai.TextDelta(block_id=f"progress-{step}", chunk=step)
        yield ai.TextEnd(block_id=f"progress-{step}")
        await asyncio.sleep(0.3)

    # The final yielded message's text is returned as the tool result.
    final = ai.Message(
        role="assistant",
        parts=[ai.TextPart(text="The mothership says: Soon.")],
    )
    yield final


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[talk_to_mothership])

    messages = [
        ai.system_message("Use the mothership tool when asked about the future."),
        ai.user_message("When will the robots take over?"),
    ]

    async for event in my_agent.run(model, messages):
        if isinstance(event, ai.TextDelta):
            if event.block_id.startswith("progress-"):
                print(f"  [{event.chunk}]")
            else:
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
