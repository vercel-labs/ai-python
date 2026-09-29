"""Pick up a conversation you did not start.

Sessions belong to the harness, not to this process — including ones you
started in your own terminal. If that terminal session is still open, its
TUI holds no SDK claim, so `resume` would make two writers: `fork` it instead
(same history, a conversation of your own). Another SDK client's open
conversation is different — `resume` refuses it with `SessionBusyError`.
"""

import asyncio

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def main() -> None:
    async with harnesses.claude_code(workspace=workspaces.Local(".")) as agent:
        for info in (await agent.sessions())[:5]:
            print(f"{info.session_id}  {info.title or ''}"[:90])

        sessions = await agent.sessions()
        if not sessions:
            print("no conversations in this directory yet")
            return

        # Read one without touching it.
        messages = await agent.history(sessions[0].session_id)
        print(f"\n{len(messages)} messages in the most recent conversation")

        # Branch it, so whoever else has it open is unaffected.
        branch = await agent.fork(sessions[0].session_id)
        answer = await branch.run("In one sentence: what were we doing?")
        print(answer.text)


asyncio.run(main())
