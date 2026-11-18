"""An ai.Agent gives its Python tools to a CLI, through the MCP bridge."""

import asyncio

import ai


@ai.tool
async def secret_word(topic: str) -> str:
    """Return the secret word for a topic."""
    return {"birds": "HERON"}.get(topic, "UNKNOWN")


async def main() -> None:
    agent = ai.Agent(tools=[secret_word])
    messages = [
        ai.user_message(
            "Use the secret_word tool for the topic 'birds'. "
            "Reply with the word only."
        )
    ]
    async with ai.external.Fx() as harness, agent.run(harness, messages) as run:
        async for event in run:
            match event:
                case ai.events.ToolEnd():  # an ai tool: the agent runs it
                    print(f"[ai tool] {event.tool_call.tool_name}")
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
    print()
    print([message.role for message in run.messages])


asyncio.run(main())
