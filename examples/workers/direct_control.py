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
    worker = KINDS[kind]()
    async with worker.session() as session:
        await session.send(
            ai.user_message("Count from 1 to 20 slowly, one number per line.")
        )

        steered = False
        async for event in session.events():
            match event:
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
                    if not steered and "5" in event.chunk:
                        steered = True
                        # A steer: the CLI decides how to fit it in.
                        await session.send(
                            ai.user_message("Stop at 8 instead.")
                        )
                case workers.MessageEvent(message=message):
                    print(f"\n[{message.role}] {message.text[:40]!r}")
                case ai.events.StreamEnd(finish_reason="stop"):
                    break

        await session.send(
            ai.user_message("Now count from 100 down to 1, slowly.")
        )
        await asyncio.sleep(3)
        await session.interrupt()  # stop the turn; the session stays
        # The cut reply is in the conversation, with the reason in its data.
        print(f"\n{session.messages[-1].provider_metadata}")


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
