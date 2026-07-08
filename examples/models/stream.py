"""Basic streaming — print text deltas as they arrive."""

import asyncio

import ai

model = ai.get_model("anthropic/claude-sonnet-4.6")

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def main() -> None:
    async with ai.stream(model, messages) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
