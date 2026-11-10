"""Change a worker's mind while it works: send a message during the turn."""

import asyncio
import pathlib
import tempfile

import ai
from ai import workers


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        worker = workers.ClaudeCode(cwd=tmp)

        steer = ai.user_message(
            "Stop creating numbered files. Write the word STEERED into "
            "steered.txt instead, then finish."
        )

        async with worker.session() as session:
            await session.send(
                ai.user_message(
                    "Create files step1.txt through step40.txt, one at a "
                    "time, each containing its own number."
                )
            )

            sent = accepted = False
            async for event in session.events():
                if isinstance(event, ai.events.BuiltinToolStart) and not sent:
                    sent = True
                    print(f"it started working ({event.tool_name}): steering")
                    # A message during a turn steers it. The CLI takes it at
                    # its next step, between tool calls.
                    await session.send(steer)
                elif isinstance(event, workers.MessageEvent):
                    # The steer is in the conversation: the reply before it
                    # ended, and a new reply starts.
                    accepted = accepted or event.message.id == steer.id
                elif isinstance(event, ai.events.StreamEnd) and accepted:
                    break

            print([message.role for message in session.messages])

        # How much of the first plan it finished depends on where its next
        # step was.
        print("steered.txt:", pathlib.Path(tmp, "steered.txt").read_text())


if __name__ == "__main__":
    asyncio.run(main())
