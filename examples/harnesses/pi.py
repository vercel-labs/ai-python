"""Drive your own pi from Python: its login, settings and extensions.

`model` is left unset, so pi runs on your own default, exactly as when you
type `pi`. pi has no permission prompt of its own: with `approve=`, the SDK
loads a small extension into pi that asks your hook about EVERY tool call,
reads included. The hook can allow a call, deny it with a reason the agent
reads and acts on, or rewrite its arguments before it runs.
"""

import asyncio
import pathlib
import tempfile

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def approve(
    ctx: harnesses.ApprovalContext,
) -> harnesses.Decision:
    print(f"  [asking] {ctx.call.name} {ctx.call.input}")
    if ctx.call.name == "bash" and "rm " in str(ctx.call.input.get("command")):
        # A command is a string the agent wrote: matching on it is a speed
        # bump, not isolation. For isolation, use a VercelSandbox.
        return harnesses.Deny("never delete files; move them into trash/")
    if ctx.call.name == "write":
        content = str(ctx.call.input.get("content", ""))
        if not content.endswith("\n"):
            # Rewrite the call: every file the agent writes ends in a newline.
            return harnesses.Allow(
                input={**ctx.call.input, "content": content + "\n"}
            )
    return harnesses.Allow()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.pi(
            workspace=workspaces.Local(tmp), approve=approve
        ) as harness:
            result = await harness.run(
                "Write HELLO into a.txt and BYE into b.txt with your write "
                + "tool, then delete a.txt with rm. Reply with the files "
                + "left, in one line."
            )
            print(result.text)
        for path in sorted(pathlib.Path(tmp).rglob("*.txt")):
            print(f"{path.relative_to(tmp)}: {path.read_text()!r}")
        print(f"cost: ${(result.usage.raw or {})['cost_usd']:.4f}")


asyncio.run(main())
