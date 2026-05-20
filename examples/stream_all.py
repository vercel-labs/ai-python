"""Streaming across all available adapter"""

import asyncio

import ai

MODELS: list[tuple[str, ai.Provider, str]] = [
    ("ai_gateway", ai.get_provider("vercel"), "anthropic/claude-sonnet-4.6"),
    ("anthropic", ai.get_provider("anthropic"), "claude-sonnet-4-6"),
    ("openai", ai.get_provider("openai"), "gpt-5.5"),
]

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def _run(name: str, provider: ai.Provider, model_id: str) -> None:
    print(f"\n{name} / {model_id}")

    if not provider.is_configured():
        print(f"[SKIP] {provider.name} provider is not configured")
        return

    model = ai.Model(model_id, provider=provider)

    try:
        async with ai.stream(model, messages) as s:
            async for event in s:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    except Exception as exc:
        print(f"[ERR] {type(exc).__name__}: {exc}")


async def main() -> None:
    for name, provider, model_id in MODELS:
        await _run(name, provider, model_id)


if __name__ == "__main__":
    asyncio.run(main())
