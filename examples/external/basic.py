"""Simplest external harness: one prompt to a CLI, where a model goes."""

import asyncio

import ai


async def main() -> None:
    messages = [ai.user_message("Reply with exactly: READY")]

    async with (
        ai.external.ClaudeCode() as harness,
        ai.stream(harness, messages) as stream,
    ):
        async for event in stream:
            match event:
                case ai.events.TextDelta():
                    print(event.chunk, end="", flush=True)
                case ai.events.StreamEnd():
                    print(f"\nfinish: {event.finish_reason}")

    if stream.usage is not None:
        print(
            f"tokens: {stream.usage.input_tokens} in / "
            f"{stream.usage.output_tokens} out"
        )


if __name__ == "__main__":
    asyncio.run(main())
