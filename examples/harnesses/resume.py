"""Stop, persist a handle, and continue in another process."""

import asyncio
import tempfile

from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp)
        ) as agent:
            session = agent.session()
            await session.run("Remember this codeword: PLATYPUS. Reply OK.")
            saved = (
                session.handle.model_dump_json()
            )  # put this in your database

        # ... later, another process entirely ...
        async with harnesses.harness_from(
            harnesses.Handle.model_validate_json(saved)
        ) as session:
            answer = await session.run(
                "What codeword did I give you? Reply with the word only."
            )
            print(answer.text)


asyncio.run(main())
