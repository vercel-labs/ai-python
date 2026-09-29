"""Change an agent's mind while it is working.

Steering is absorbed at the agent's next step — between tool calls, not
mid-sentence — so it redirects work in progress rather than interrupting a
single long answer.
"""

import asyncio
import tempfile

from ai.harnesses import experimental as harnesses
from ai.types import events
from ai.workspaces import experimental as workspaces


async def approve(
    ctx: harnesses.ApprovalContext,
) -> harnesses.Decision:
    return harnesses.Allow()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp), approve=approve
        ) as agent:
            session = agent.session()
            turn = session.stream(
                "Create files step1.txt through step40.txt, one at a time, "
                + "each containing its own number."
            )

            steered = False
            async for event in turn:
                if isinstance(event, events.ToolStart) and not steered:
                    steered = True
                    print(
                        f"it started working ({event.tool_name}) — redirecting"
                    )
                    await session.steer(
                        "Stop creating numbered files. Write the word STEERED "
                        + "into steered.txt instead, then finish."
                    )

            # It took the new instruction. How much of the original plan it
            # finished first depends on where its next step boundary fell.
            print(
                "steered.txt:", await agent.workspace.read_text("steered.txt")
            )


asyncio.run(main())
