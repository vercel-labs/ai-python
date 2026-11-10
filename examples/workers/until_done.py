"""Your own loop around a worker: prompt, check, prompt again."""

import asyncio
import dataclasses
import subprocess
import tempfile
from collections.abc import AsyncGenerator

import ai
from ai import workers

EXPECTED = (
    "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz"
)


@dataclasses.dataclass
class Checked:
    attempt: int
    passed: bool
    detail: str


# The CLI owns the loop inside a turn. The loop around the turns is plain
# Python: run a turn, check the result in the directory, and send the failure
# back until the check passes.
async def until_passes(
    worker: workers.Worker, task: str, *, attempts: int = 3
) -> AsyncGenerator[ai.events.Event | Checked]:
    async with worker.session() as session:
        prompt = task
        for attempt in range(1, attempts + 1):
            messages = [*session.messages, ai.user_message(prompt)]
            async with ai.stream(worker, messages) as stream:
                async for event in stream:
                    if not isinstance(event, ai.events.ReasoningDelta):
                        yield event

            run = await asyncio.to_thread(
                lambda: subprocess.run(
                    ["python3", "fizzbuzz.py"],
                    cwd=worker.cwd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            )
            output = run.stdout.strip()
            passed = run.returncode == 0 and output == EXPECTED
            yield Checked(attempt, passed, output or run.stderr.strip())
            if passed:
                return

            prompt = (
                "`python3 fizzbuzz.py` should print exactly:\n"
                + EXPECTED
                + f"\n\nIt printed (exit code {run.returncode}):\n"
                + (output or run.stderr.strip())
                + "\n\nFix fizzbuzz.py."
            )


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        worker = workers.ClaudeCode(cwd=tmp)

        # Deliberately vague, so the check has something to catch.
        task = "Write fizzbuzz.py: a FizzBuzz program."

        async for item in until_passes(worker, task):
            if isinstance(item, Checked):
                verdict = "passed" if item.passed else "failed"
                print(f"check {item.attempt}: {verdict}")
            elif isinstance(item, ai.events.BuiltinToolEnd):
                print(f"  [worker] {item.tool_call.tool_name}")


if __name__ == "__main__":
    asyncio.run(main())
