"""Explicit provider — use a local OpenAI-compatible server."""

import asyncio
import os

import ai

messages = [ai.user_message("Hello!")]


async def main() -> None:
    # Example for local OpenAI-compatible servers like LM Studio.
    provider = ai.get_provider(
        "openai",
        base_url=os.environ.get(
            "LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1"
        ),
        api_key=os.environ.get("LOCAL_OPENAI_API_KEY", "some-key"),
        headers={"X-Custom-Header": "example"},
    )

    model = ai.Model(
        os.environ.get("LOCAL_OPENAI_MODEL", "local-model"),
        provider=provider,
    )

    try:
        try:
            await ai.probe(model)
        except ai.ProviderError as exc:
            print(
                f"[SKIP] local OpenAI-compatible server is unavailable: {exc}"
            )
            return

        async with ai.stream(model, messages) as s:
            async for event in s:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    finally:
        # Explicit providers need explicit cleanup.
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
