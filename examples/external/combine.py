"""Run two external harnesses at the same time and read them as one stream."""

import asyncio
import tempfile
from collections.abc import AsyncGenerator

import ai


async def tagged(
    label: str, harness: ai.external.ExternalHarness, prompt: str
) -> AsyncGenerator[tuple[str, ai.events.Event]]:
    async with (
        harness.session(),
        ai.stream(harness, [ai.user_message(prompt)]) as stream,
    ):
        async for event in stream:
            yield label, event


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # An external harness is one conversation: two at a time are two
        # external harnesses.
        poem = ai.external.ClaudeCode(cwd=tmp)
        limerick = ai.external.ClaudeCode(cwd=tmp)

        async with ai.util.merge(
            tagged(
                "poem",
                poem,
                "Write a two-line poem about rivers into poem.txt.",
            ),
            tagged(
                "limerick",
                limerick,
                "Write a limerick about mountains into limerick.txt.",
            ),
        ) as events:
            async for label, event in events:
                match event:
                    case ai.events.BuiltinToolEnd():
                        print(f"[{label}] {event.tool_call.tool_name}")
                    case ai.events.StreamEnd():
                        print(f"[{label}] done ({event.finish_reason})")
                        print(f"[{label}] {event.message.text.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
