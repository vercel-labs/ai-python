"""Run one prompt and read the result."""

import asyncio

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def main() -> None:
    async with harnesses.claude_code(workspace=workspaces.Local(".")) as agent:
        result = await agent.run("Reply with exactly: READY")

        print(result.text)
        print(f"finish: {result.finish_reason}")
        print(
            f"tokens: {result.usage.input_tokens} in / "
            f"{result.usage.output_tokens} out"
        )


asyncio.run(main())
