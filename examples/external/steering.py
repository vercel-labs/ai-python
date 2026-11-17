"""Change an external harness's mind while it works: send during the turn."""

import asyncio
import pathlib
import tempfile

import ai


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        steer = ai.user_message(
            "Stop creating numbered files. Write the word STEERED into "
            "steered.txt instead, then finish."
        )

        async with ai.external.ClaudeCode(cwd=tmp) as harness:
            await harness.session.send(
                ai.user_message(
                    "Create files step1.txt through step40.txt, one at a "
                    "time, each containing its own number."
                )
            )

            sent = accepted = False
            async for event in harness.session.events():
                match event:
                    case ai.events.BuiltinToolStart() if not sent:
                        sent = True
                        print(
                            f"it started working ({event.tool_name}): steering"
                        )
                        # A message during a turn steers it. The CLI takes it
                        # at its next step, between tool calls.
                        await harness.session.send(steer)
                    case ai.external.MessageEvent():
                        # The steer is in the conversation: the reply before
                        # it ended, and a new reply starts.
                        accepted = accepted or event.message.id == steer.id
                    case ai.events.StreamEnd() if accepted:
                        break

            print([message.role for message in harness.session.messages])

        # How much of the first plan it finished depends on where its next
        # step was.
        print("steered.txt:", pathlib.Path(tmp, "steered.txt").read_text())


if __name__ == "__main__":
    asyncio.run(main())
