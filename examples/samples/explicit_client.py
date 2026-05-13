"""Explicit client — bring your own auth and base URL."""

import asyncio
import os

import ai

# Explicit client — useful for custom auth, proxies, or self-hosted gateways.
client = ai.Client(
    base_url="https://ai-gateway.vercel.sh/v3/ai",
    api_key=os.environ["AI_GATEWAY_API_KEY"],
    headers={"X-Custom-Header": "example"},
)

model = ai.get_model("gateway:anthropic/claude-sonnet-4", client=client)

messages = [ai.user_message("Hello!")]


async def main() -> None:
    try:
        async with ai.stream(model, messages) as s:
            async for event in s:
                if isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
        print()
    finally:
        # Explicit clients need explicit cleanup.
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
