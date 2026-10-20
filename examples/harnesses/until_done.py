"""Your own loop around a harness: prompt, check, prompt again.

The CLI owns the loop inside a turn. The loop around turns is yours, in
plain Python: here an async generator that runs a turn, checks the result in
the workspace, and sends the failure back until the check passes. It passes
the turn's events on (minus reasoning) and yields events of its own for the
checks, so the consumer sees one stream.
"""

import asyncio
import dataclasses
import tempfile
from collections.abc import AsyncIterator

import ai
from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces

EXPECTED = (
    "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz"
)


@dataclasses.dataclass
class Checked:
    attempt: int
    passed: bool
    detail: str


async def until_passes(
    harness: harnesses.Harness, task: str, *, attempts: int = 3
) -> AsyncIterator[ai.events.AgentEvent | Checked]:
    session = harness.session()
    prompt = task
    for attempt in range(1, attempts + 1):
        async with session.stream(prompt) as turn:
            async for event in turn:
                if not isinstance(event, ai.events.ReasoningDelta):
                    yield event
        run = await harness.workspace.exec(
            ["python3", "fizzbuzz.py"], timeout=30
        )
        output = run.stdout.strip()
        passed = run.exit_code == 0 and output == EXPECTED
        yield Checked(attempt, passed, output or run.stderr.strip())
        if passed:
            return
        prompt = (
            "`python3 fizzbuzz.py` should print exactly:\n"
            + EXPECTED
            + f"\n\nIt printed (exit code {run.exit_code}):\n"
            + (output or run.stderr.strip())
            + "\n\nFix fizzbuzz.py."
        )


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp)
        ) as harness:
            # Deliberately vague, so the check has something to catch.
            task = "Write fizzbuzz.py: a FizzBuzz program."
            async for item in until_passes(harness, task):
                if isinstance(item, Checked):
                    verdict = "passed" if item.passed else "failed"
                    print(f"check {item.attempt}: {verdict}")
                elif isinstance(item, ai.events.ToolEnd):
                    print(f"  [harness] {item.tool_call.tool_name}")


asyncio.run(main())
