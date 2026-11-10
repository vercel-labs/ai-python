"""Simplest worker: one prompt to a coding-agent CLI, where a model goes."""

import asyncio

import ai
from ai import workers


async def main() -> None:
    worker = workers.ClaudeCode()

    messages = [ai.user_message("Reply with exactly: READY")]

    async with ai.stream(worker, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            elif isinstance(event, ai.events.StreamEnd):
                print(f"\nfinish: {event.finish_reason}")

    if stream.usage is not None:
        print(
            f"tokens: {stream.usage.input_tokens} in / "
            f"{stream.usage.output_tokens} out"
        )


if __name__ == "__main__":
    asyncio.run(main())
