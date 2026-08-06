"""Non-streaming LLM call.

``ai.experimental_generate()`` is the counterpart of ``ai.stream()``:
same arguments, but it returns the final message instead of yielding
events.
"""

import asyncio

import ai

model = ai.get_model("anthropic/claude-sonnet-4.6")

messages = [
    ai.system_message("Be concise."),
    ai.user_message("Explain why the sky is blue in two sentences."),
]


async def main() -> None:
    message = await ai.experimental_generate(model, messages)
    print(message.text)


if __name__ == "__main__":
    asyncio.run(main())
