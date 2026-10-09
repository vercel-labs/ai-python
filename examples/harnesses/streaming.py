"""Watch a turn as it happens, and pick up each piece the moment it is whole.

A turn streams AI SDK events. You never reassemble them yourself: every
event's `.message` is the assistant message accumulated so far, and a
`ToolEnd` carries the complete call. Print deltas as they arrive for a
live feel; act on the *End events when you need the finished thing.
"""

import asyncio
import json
import tempfile

from ai.harnesses import experimental as harnesses
from ai.types import events, messages
from ai.workspaces import experimental as workspaces


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp)
        ) as agent:
            session = agent.session()
            turn = session.stream(
                "Write the word HELLO into greeting.txt, then read it back "
                + "and reply with what it contains."
            )

            async for event in turn:
                if isinstance(event, events.TextDelta):
                    # Live output: print text as it is generated.
                    print(event.chunk, end="", flush=True)

                elif isinstance(event, events.TextEnd):
                    # The whole paragraph, already assembled on the message.
                    part = next(
                        p
                        for p in event.message.parts
                        if isinstance(p, messages.TextPart)
                        and p.id == event.block_id
                    )
                    print(f"\n[text done: {len(part.text)} chars]")

                elif isinstance(event, events.ToolEnd):
                    # The complete call: native name, arguments as a JSON
                    # string.
                    args = json.loads(event.tool_call.tool_args)
                    print(f"[tool call] {event.tool_call.tool_name}({args})")

                elif isinstance(event, events.ToolCallResult):
                    # What the tool returned, as its own role="tool" message.
                    for tool_result in event.results:
                        print(
                            f"[tool result] {tool_result.tool_name or '?'}: "
                            f"{str(tool_result.result)[:80]}"
                        )

                elif isinstance(event, events.StreamEnd):
                    print(f"[turn over] finish={event.finish_reason}")

            # Everything above is also on the settled result.
            result = turn.result
            print(f"\nresult.text: {result.text.strip()!r}")
            print(
                f"tokens: {result.usage.input_tokens} in / "
                f"{result.usage.output_tokens} out"
            )
            print(f"messages in history: {len(result.messages)}")


asyncio.run(main())
