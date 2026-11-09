"""An ai.Agent gives its Python tools to a CLI, through the MCP bridge.

uv run python examples/workers/agent_tools.py [codex]
"""

import asyncio
import sys

import ai
from ai import workers

KINDS: dict[str, type[workers.Worker]] = {
    "codex": workers.Codex,
}


@ai.tool
async def secret_word(topic: str) -> str:
    """Return the secret word for a topic."""
    return {"birds": "HERON"}.get(topic, "UNKNOWN")


async def main(kind: str) -> None:
    agent = ai.Agent(tools=[secret_word])
    messages = [
        ai.user_message(
            "Use the secret_word tool for the topic 'birds'. "
            "Reply with the word only."
        )
    ]
    async with KINDS[kind]() as worker:
        async with agent.run(worker, messages) as run:
            async for event in run:
                match event:
                    case ai.events.ToolEnd():  # an ai tool: the agent runs it
                        print(f"[ai tool] {event.tool_call.tool_name}")
                    case ai.events.TextDelta():
                        print(event.chunk, end="", flush=True)
        print()
        print([message.role for message in worker.messages])


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
