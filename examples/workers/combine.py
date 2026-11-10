"""Run two workers at the same time and read them as one stream."""

import asyncio
import tempfile
from collections.abc import AsyncGenerator

import ai
from ai import workers


async def tagged(
    label: str, worker: workers.Worker, prompt: str
) -> AsyncGenerator[tuple[str, ai.events.Event]]:
    async with ai.stream(worker, [ai.user_message(prompt)]) as stream:
        async for event in stream:
            yield label, event


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # A worker is one conversation: two at a time are two workers.
        poem = workers.ClaudeCode(cwd=tmp)
        limerick = workers.ClaudeCode(cwd=tmp)

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
                if isinstance(event, ai.events.BuiltinToolEnd):
                    print(f"[{label}] {event.tool_call.tool_name}")
                elif isinstance(event, ai.events.StreamEnd):
                    print(f"[{label}] done ({event.finish_reason})")
                    print(f"[{label}] {event.message.text.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
