"""Give an agent a project, then collect what it produced.

`copy` moves a tree either way. Direction is whichever side is marked with
`workspace / "path"` — there is no flag, and no second method.
"""

import asyncio
import tempfile
from pathlib import Path

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def approve(
    ctx: harnesses.ApprovalContext,
) -> harnesses.Decision:
    return harnesses.Allow()


async def main() -> None:
    project = Path(__file__).parents[2] / "src"

    with tempfile.TemporaryDirectory() as collected:
        # The gateway is the workspace's: the key is injected at egress, and
        # nothing else from .env.local leaves this machine.
        async with workspaces.VercelSandbox(
            gateway=workspaces.vercel_ai_gateway()
        ) as workspace:
            # In: the tree minus what .gitignore excludes, batched into few
            # requests.
            sent = await workspaces.copy(project, workspace / "src")
            print(f"sent {sent} files into the sandbox")

            async with harnesses.claude_code(
                workspace=workspace, approve=approve
            ) as agent:
                await agent.run(
                    "Read src/ai/harnesses/experimental/errors.py and write a "
                    "one-paragraph "
                    + "summary to notes/errors.md. Then reply DONE."
                )

            # Out: one archive, one round trip, however many files.
            back = await workspaces.copy(workspace / "notes", collected)
            print(f"brought back {back} files")

        print((Path(collected) / "errors.md").read_text()[:200])


asyncio.run(main())
