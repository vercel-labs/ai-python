"""Watch a worker's turn as it happens, then continue the conversation."""

import asyncio
import tempfile

import ai
from ai import workers


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        worker = workers.ClaudeCode(cwd=tmp)

        messages = [
            ai.user_message(
                "Write the word HELLO into greeting.txt, then read it back "
                "and reply with what it contains."
            )
        ]

        async with ai.stream(worker, messages) as stream:
            async for event in stream:
                if isinstance(event, ai.events.TextDelta):
                    # Live output: print text as it is generated.
                    print(event.chunk, end="", flush=True)
                elif isinstance(event, ai.events.TextEnd):
                    # The whole block, already assembled on the message.
                    part = next(
                        p
                        for p in event.message.parts
                        if isinstance(p, ai.messages.TextPart)
                        and p.id == event.block_id
                    )
                    print(f"\n[text done: {len(part.text)} chars]")
                elif isinstance(event, ai.events.BuiltinToolEnd):
                    # The CLI's own tool: the complete call, with the native
                    # name and the arguments as a JSON string.
                    call = event.tool_call
                    print(f"[tool call] {call.tool_name} {call.tool_args[:60]}")
                elif isinstance(event, ai.events.BuiltinToolResult):
                    # The CLI ran the tool: the result is a part of the same
                    # assistant message, not a tool message.
                    result = event.result
                    print(
                        f"[tool result] {result.tool_name}: "
                        f"{str(result.result)[:80]}"
                    )
                elif isinstance(event, ai.events.StreamEnd):
                    print(f"[turn over] finish={event.finish_reason}")

        # Everything above is also on the message.
        print(f"\ntext: {stream.text.strip()!r}")
        print(f"usage: {stream.usage}")
        print(f"parts in the message: {len(stream.message.parts)}")

        # As with a model: send the history back with what is new. The
        # worker continues the same CLI session.
        messages = [
            *messages,
            stream.message,
            ai.user_message(
                "Which file did you write? Reply with its name only."
            ),
        ]
        print((await ai.experimental_generate(worker, messages)).text)


if __name__ == "__main__":
    asyncio.run(main())
