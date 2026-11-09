"""A worker where a model goes: stream it, then continue the conversation.

uv run python examples/workers/streaming.py [claude-code|codex|pi]
"""

import asyncio
import sys

import ai
from ai import workers

KINDS: dict[str, type[workers.Worker]] = {
    "claude-code": workers.ClaudeCode,
    "codex": workers.Codex,
    "pi": workers.Pi,
}


async def main(kind: str) -> None:
    async with KINDS[kind]() as worker:
        messages = [
            ai.user_message(
                "List the files here with a shell command, then say DONE."
            )
        ]
        async with ai.stream(worker, messages) as s:
            async for event in s:
                match event:
                    case ai.events.TextDelta():
                        print(event.chunk, end="", flush=True)
                    case ai.events.BuiltinToolEnd():  # the CLI's own tool
                        call = event.tool_call
                        print(f"\n[{call.tool_name}] {call.tool_args[:60]}")
        print(f"\nusage: {s.message.usage}")

        # The worker keeps the conversation: send it back with what is new.
        follow_up = [
            *worker.messages,
            ai.user_message("How many files? Reply with a number only."),
        ]
        print((await ai.experimental_generate(worker, follow_up)).text)

    # A worker is data: save it, and continue the conversation later.
    saved = worker.model_dump_json()
    again = workers.Worker.model_validate_json(saved)
    async with again:
        reply = await ai.experimental_generate(
            again, [*again.messages, ai.user_message("And the largest one?")]
        )
        print(reply.text)


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
