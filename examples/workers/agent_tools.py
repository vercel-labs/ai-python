"""A worker as the model of an `ai.Agent`: the agent's tools reach it by MCP.

The CLI calls `lookup_order`. The worker hands the call to the agent, the
agent runs the tool, and the CLI gets the result and goes on.

    uv run python examples/workers/agent_tools.py [codex]
"""

import asyncio
import sys
from collections.abc import Callable

import ai
from ai import workers

KINDS: dict[str, Callable[..., workers.Worker]] = {
    "codex": workers.Codex,
}


@ai.tool
async def lookup_order(order_id: str) -> str:
    """Look up the status of an order by its id."""
    return f"order {order_id}: shipped on 2026-11-02"


async def main(kind: str) -> None:
    agent = ai.Agent(tools=[lookup_order])
    messages = [
        ai.user_message(
            "Use the lookup_order tool for order A-17, then tell me its "
            "status in one sentence."
        )
    ]
    async with KINDS[kind]() as worker:
        async with agent.run(worker, messages) as run:
            async for event in run:
                match event:
                    case ai.events.ToolEnd():  # the agent runs this one
                        call = event.tool_call
                        print(f"[agent tool] {call.tool_name} {call.tool_args}")
                    case ai.events.BuiltinToolEnd():  # the CLI's own tool
                        print(f"[cli tool] {event.tool_call.tool_name}")
                    case ai.events.TextDelta():
                        print(event.chunk, end="", flush=True)
        print()
        print([m.role for m in run.messages])


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
