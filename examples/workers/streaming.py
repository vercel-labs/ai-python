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
    worker = KINDS[kind]()
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

    # As with a model: send the history back with what is new. The worker
    # continues the same CLI session.
    messages = [
        *messages,
        s.message,
        ai.user_message("How many files? Reply with a number only."),
    ]
    print((await ai.experimental_generate(worker, messages)).text)

    # The worker remembers its session: save it, and continue later. A
    # session block keeps the CLI alive between the calls.
    again = workers.Worker.model_validate_json(worker.model_dump_json())
    async with again.session() as session:
        for question in ("Name one file here.", "And the largest one?"):
            messages = [*session.messages, ai.user_message(question)]
            print((await ai.experimental_generate(again, messages)).text)


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "codex"))
