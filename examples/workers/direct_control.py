"""Direct control: send, steer, read the feed, interrupt.

uv run python examples/workers/direct_control.py [claude-code|codex|pi]
"""

import asyncio
import sys

import ai
from ai import workers

KINDS: dict[str, type[workers.Worker]] = {
    "claude-code": workers.ClaudeCode,
    "codex": workers.Codex,
    "pi": workers.Pi,
}


async def main(kind: str) -> None:
    async with KINDS[kind]() as worker:
        feed = worker.events()  # subscribe before send: see everything
        await worker.send("Count from 1 to 20 slowly, one number per line.")

        steered = False
        async for event in feed:
            match event:
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
                    if not steered and "5" in event.chunk:
                        steered = True
                        # A steer: the CLI decides how to fit it in.
                        await worker.send("Stop at 8 instead.")
                case workers.MessageEvent(message=message):
                    print(f"\n[{message.role}] {message.text[:40]!r}")
                case ai.events.StreamEnd(finish_reason="stop"):
                    break

        await worker.send("Now count from 100 down to 1, slowly.")
        await asyncio.sleep(3)
        await worker.interrupt()  # stop the turn; the session stays
        print(f"\ninterrupted: {workers.is_interrupted(worker.messages[-1])}")
        await feed.aclose()


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
