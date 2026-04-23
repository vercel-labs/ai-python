import ai
import asyncio


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-opus-4.7")

    messages = [
        ai.system_message("you are a coding assistant"),
        ai.user_message("actually i don't need assistance thanks"),
    ]

    async for e in ai.stream(model, messages):
        print(e)


if __name__ == "__main__":
    asyncio.run(main())
