"""Approve each of the CLI's own tool calls, as `ai.Agent` tool approvals.

With `approve=True`, every tool call the CLI makes becomes a `ToolApproval`
hook. Answer it with `ai.resolve_hook`, the way any tool approval is.

    uv run python examples/workers/approvals.py [codex]
"""

import asyncio
import sys
from collections.abc import Callable

import ai
from ai import workers

KINDS: dict[str, Callable[..., workers.Worker]] = {
    "codex": workers.Codex,
}


async def main(kind: str) -> None:
    messages = [ai.user_message("Create notes.txt containing 'hello'.")]
    async with KINDS[kind](approve=True) as worker:
        async with ai.Agent().run(worker, messages) as run:
            async for event in run:
                match event:
                    case ai.events.PendingHookEvent(hook=hook):
                        meta = hook.metadata
                        answer = await asyncio.to_thread(
                            input,
                            f"allow {meta['tool']} {meta['kwargs']}? [y/N] ",
                        )
                        granted = answer.strip().lower() == "y"
                        ai.resolve_hook(
                            hook, {"granted": granted, "reason": "user said no"}
                        )
                    case ai.events.TextDelta():
                        print(event.chunk, end="", flush=True)
        print()


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
