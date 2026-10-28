"""Run two turns at once and read their events as one stream.

Each session runs one turn at a time, so parallel work takes one session
per task. `ai.util.merge` interleaves their events as they arrive; tag each
stream first if you need to know which one an event came from.
"""

import asyncio
import tempfile
from collections.abc import AsyncIterator

import ai
from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces


async def tagged(
    label: str, turn: harnesses.Turn
) -> AsyncIterator[tuple[str, ai.events.AgentEvent]]:
    async with turn:
        async for event in turn:
            yield label, event


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp)
        ) as harness:
            poem = harness.session().stream(
                "Write a two-line poem about rivers into poem.txt."
            )
            limerick = harness.session().stream(
                "Write a limerick about mountains into limerick.txt."
            )

            async for label, event in ai.util.merge(
                tagged("poem", poem), tagged("limerick", limerick)
            ):
                if isinstance(event, ai.events.ToolEnd):
                    print(f"[{label}] {event.tool_call.tool_name}")
                elif isinstance(event, ai.events.StreamEnd):
                    print(f"[{label}] done ({event.finish_reason})")

            print(f"poem: {poem.output}")
            print(f"limerick: {limerick.output}")


asyncio.run(main())
