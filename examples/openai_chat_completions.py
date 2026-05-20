"""OpenAI Chat Completions protocol — stream text from GPT-5.5."""

import asyncio

import ai
from ai.providers.openai import OpenAIChatCompletionsProtocol

messages = [
    ai.system_message("Be concise."),
    ai.user_message(
        "Explain what the OpenAI Chat Completions API is in two sentences."
    ),
]


async def main() -> None:
    provider = ai.get_provider("openai")
    if not provider.is_configured():
        print(f"[SKIP] {provider.name} provider is not configured")
        return

    model = ai.Model("gpt-5.5", provider=provider)

    try:
        async with ai.stream(
            model,
            messages,
            protocol=OpenAIChatCompletionsProtocol(),
        ) as stream:
            async for event in stream:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
