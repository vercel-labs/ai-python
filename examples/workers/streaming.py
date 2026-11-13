"""Watch a worker's turn as it happens, then continue the conversation."""

import asyncio
import tempfile

import ai


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        worker = ai.workers.ClaudeCode(cwd=tmp)

        messages = [
            ai.user_message(
                "Write the word HELLO into greeting.txt, then read it back "
                "and reply with what it contains."
            )
        ]

        async with ai.stream(worker, messages) as stream:
            async for event in stream:
                match event:
                    case ai.events.TextDelta():
                        # Live output: print text as it is generated.
                        print(event.chunk, end="", flush=True)
                    case ai.events.TextEnd(message=message, block_id=block_id):
                        # The whole block, already assembled on the message.
                        part = next(
                            p
                            for p in message.parts
                            if isinstance(p, ai.messages.TextPart)
                            and p.id == block_id
                        )
                        print(f"\n[text done: {len(part.text)} chars]")
                    case ai.events.BuiltinToolEnd(tool_call=call):
                        # The CLI's own tool: the complete call, with the
                        # native name and the arguments as a JSON string.
                        print(
                            f"[tool call] {call.tool_name} "
                            f"{call.tool_args[:60]}"
                        )
                    case ai.events.BuiltinToolResult(result=result):
                        # The CLI ran the tool: the result is a part of the
                        # same assistant message, not a tool message.
                        print(
                            f"[tool result] {result.tool_name}: "
                            f"{str(result.result)[:80]}"
                        )
                    case ai.events.StreamEnd():
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
