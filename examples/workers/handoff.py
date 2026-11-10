"""Hand a conversation from one coding-agent CLI to a different one."""

import asyncio
import pathlib
import tempfile

import ai
from ai import workers


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # 1. A conversation with Claude Code. It does real work.
        claude = workers.ClaudeCode(cwd=tmp)

        messages = [
            ai.user_message(
                "Create notes.txt containing exactly: milk, eggs. Also "
                "remember: the deadline is FRIDAY. Reply with exactly: OK"
            )
        ]
        reply = await ai.experimental_generate(claude, messages)
        print(
            f"Claude Code: {reply.text.strip()} "
            f"({len(reply.builtin_tool_calls)} tool call(s))"
        )

        # 2. Codex continues it, in the same directory. As with a model:
        #    give the history to the new worker. It starts a new CLI session
        #    with that past, thus it knows what was said and what was done.
        codex = workers.Codex(cwd=tmp)

        messages = [
            *messages,
            reply,
            ai.user_message(
                "What is the deadline, and what did we put in notes.txt? "
                "Then append 'bread' to notes.txt. Reply in one line."
            ),
        ]
        reply = await ai.experimental_generate(codex, messages)
        print("Codex:", reply.text.strip())

        notes = pathlib.Path(tmp, "notes.txt").read_text()
        print("notes.txt:", notes.strip().replace("\n", " / "))


if __name__ == "__main__":
    asyncio.run(main())
