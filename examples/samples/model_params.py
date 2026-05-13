"""Model params — request-scoped provider options."""

import asyncio

import ai

model = ai.get_model("gateway:anthropic/claude-sonnet-4")

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why waterfalls look white in two sentences."),
]


async def main() -> None:
    params = {
        "providerOptions": {
            "gateway": {"sort": "cost"},
            "anthropic": {"speed": "fast"},
        }
    }
    async with ai.stream(model, messages, params=params) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
