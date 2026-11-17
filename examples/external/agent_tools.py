"""An ai.Agent gives its Python tools to a CLI, through the MCP bridge.

uv run python examples/external/agent_tools.py [claude-code|codex|fx|pi]
"""

import asyncio
import sys

import ai

KINDS: dict[str, type[ai.external.ExternalHarness]] = {
    "claude-code": ai.external.ClaudeCode,
    "codex": ai.external.Codex,
    "fx": ai.external.Fx,
    "pi": ai.external.Pi,
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
    harness = KINDS[kind]()
    async with harness.session(), agent.run(harness, messages) as run:
        async for event in run:
            match event:
                case ai.events.ToolEnd():  # an ai tool: the agent runs it
                    print(f"[ai tool] {event.tool_call.tool_name}")
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
    print()
    print([message.role for message in run.messages])


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
