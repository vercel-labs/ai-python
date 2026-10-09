"""Decide what the agent is allowed to do, in your own code.

Without a hook every tool call runs: nobody is at a terminal to answer.
With one, it asks you — and a denial's reason is delivered to the agent.
"""

import asyncio
import tempfile

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def approve(
    ctx: harnesses.ApprovalContext,
) -> harnesses.Decision:
    print(f"  [asking] {ctx.call.name} {ctx.call.input}")
    target = ctx.call.input.get("file_path", "")
    if target and not ctx.workspace.contains(target):
        return harnesses.Deny("that path is outside the workspace")
    return harnesses.Allow()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp), approve=approve
        ) as agent:
            result = await agent.run(
                "Write the word HELLO into greeting.txt, then reply DONE."
            )
            print(result.text)
            print(
                "greeting.txt says:",
                await agent.workspace.read_text("greeting.txt"),
            )


asyncio.run(main())
