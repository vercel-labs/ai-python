"""Streaming through AI Gateway."""

import asyncio

import ai

model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def main() -> None:
    provider = model.provider
    if not provider.is_configured():
        print(f"[SKIP] {provider.name} provider is not configured")
        return

    async with ai.stream(model, messages) as s:
        async for event in s:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
