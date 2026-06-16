"""Streaming across all available adapter"""

import asyncio

import ai

MODELS: list[tuple[str, ai.Model]] = [
    ("ai_gateway", ai.get_model("gateway:anthropic/claude-sonnet-4.6")),
    ("anthropic", ai.get_model("anthropic:claude-sonnet-4-6")),
    ("openai", ai.get_model("openai:gpt-5.5")),
]

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def _run(name: str, model: ai.Model) -> None:
    print(f"\n{name} / {model.id}")

    if not model.provider.is_configured():
        print(f"[SKIP] {model.provider.name} provider is not configured")
        return

    try:
        async with ai.stream(model, messages) as s:
            async for event in s:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    except Exception as exc:
        print(f"[ERR] {type(exc).__name__}: {exc}")


async def main() -> None:
    for name, model in MODELS:
        await _run(name, model)


if __name__ == "__main__":
    asyncio.run(main())
