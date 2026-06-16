"""OpenAI Chat Completions protocol — stream text from GPT-5.5."""

import asyncio

import ai

messages = [
    ai.system_message("Be concise."),
    ai.user_message(
        "Explain what the OpenAI Chat Completions API is in two sentences."
    ),
]


async def main() -> None:
    model = ai.get_model("openai:gpt-5.5").with_protocol(
        "openai.chat_completions"
    )
    if not model.provider.is_configured():
        print(f"[SKIP] {model.provider.name} provider is not configured")
        return

    try:
        async with ai.stream(model, messages) as stream:
            async for event in stream:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
