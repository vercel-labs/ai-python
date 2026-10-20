"""Decide what the agent is allowed to do, in your own code.

Without a hook every tool call runs: nobody is at a terminal to answer.
With one, it asks you — and a denial's reason is delivered to the agent.

One hook for both CLIs: Claude Code calls its shell `Bash` and its file
tools take `file_path`; Codex calls its shell `commandExecution` and its
edits `fileChange`.
"""

import asyncio
import tempfile

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces

SHELL = {"Bash", "commandExecution"}


async def approve(
    ctx: harnesses.ApprovalContext,
) -> harnesses.Decision:
    print(f"  [asking] {ctx.call.name} {ctx.call.input}")
    if ctx.call.name in SHELL:
        # A command is a string the agent wrote: matching on it is a speed
        # bump, not isolation. For isolation, use a VercelSandbox.
        command = str(ctx.call.input.get("command", ""))
        if "rm -rf" in command or "git push" in command:
            return harnesses.Deny("not that command")
        return harnesses.Allow()
    target = ctx.call.input.get("file_path")
    if target:
        # Resolve first: a symlink inside the workspace can point out of it.
        real = await ctx.workspace.resolve(target)
        if not ctx.workspace.contains(real):
            return harnesses.Deny("that path is outside the workspace")
    return harnesses.Allow()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp), approve=approve
        ) as harness:
            result = await harness.run(
                "Write the word HELLO into greeting.txt, then reply DONE."
            )
            print(result.text)
            print(
                "greeting.txt says:",
                await harness.workspace.read_text("greeting.txt"),
            )


asyncio.run(main())
