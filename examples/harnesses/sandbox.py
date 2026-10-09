"""The same code, with the agents running in a Vercel Sandbox microVM.

Only the workspace changes. The CLI is installed into the VM for you.

`vercel link` + `vercel env pull` is enough to CREATE a sandbox —
VercelSandbox reads the Vercel credentials out of .env.local. Nothing else
in that file leaves your machine unless you say so.

What a harness needs to reach a MODEL is a separate question, because it
authenticates where it runs and a fresh VM has no login. vercel_ai_gateway()
answers it, and it goes to the WORKSPACE: the sandbox locks its egress to
the gateway host and writes the key into requests on the way out, and every
harness opened on it is handed a placeholder. The key never enters the VM.
"""

import asyncio

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def main() -> None:
    gw = workspaces.vercel_ai_gateway()  # reads AI_GATEWAY_API_KEY

    async with workspaces.VercelSandbox(gateway=gw) as workspace:
        await workspace.write_text(
            "util.py", "def add(a, b):\n    return a + b\n"
        )

        async with harnesses.claude_code(workspace=workspace) as agent:
            print("running", agent.version, "in the sandbox")
            result = await agent.run(
                "Read util.py and describe the code in one line."
            )
            print(result.text.strip())

        async with harnesses.codex(workspace=workspace) as agent:
            print("running", agent.version, "in the sandbox")
            result = await agent.run(
                "Read util.py and describe the code in one line."
            )
            print(result.text.strip())


asyncio.run(main())
