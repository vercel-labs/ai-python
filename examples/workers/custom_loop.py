"""Custom loop: a worker turn as one step of your own agent loop."""

import asyncio
import pathlib
import subprocess
import tempfile
from collections.abc import AsyncGenerator

import ai
from ai import workers

REASONING = (
    ai.events.ReasoningStart,
    ai.events.ReasoningDelta,
    ai.events.ReasoningEnd,
)


class CodeThenReview(ai.Agent):
    def __init__(self, worker: workers.Worker) -> None:
        super().__init__()
        self.worker = worker

    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[ai.events.AgentEvent]:
        """The worker does the task, then the model reviews the result."""
        # Step 1: the worker does the task. Each run is a new conversation,
        # thus a copy of the worker with no session.
        worker = self.worker.model_copy(update={"session_id": None})
        async with ai.stream(worker, [context.messages[-1]]) as work:
            async for event in work:
                if not isinstance(event, REASONING):
                    yield event

        # Step 2: evidence from the directory, not the worker's word for
        # it. Only the worker's answer joins the agent's context, not its
        # tool calls: they name tools that this model does not have.
        source = pathlib.Path(worker.cwd, "hello.py").read_text()
        run = await asyncio.to_thread(
            lambda: subprocess.run(
                ["python3", "hello.py"],
                cwd=worker.cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        )
        context.add(ai.assistant_message(work.text))
        context.add(
            ai.user_message(
                "You are the reviewer. hello.py contains:\n"
                + source
                + f"\nRunning it printed: {run.stdout.strip()!r} "
                + f"(exit code {run.returncode}).\n"
                + "In one sentence: does it meet the task? Start with YES "
                + "or NO."
            )
        )

        # Step 3: a model call, like any other agent step.
        async with ai.stream(context=context) as review:
            async for event in review:
                yield event
        context.add(review.message)


async def main() -> None:
    model = ai.get_model("anthropic/claude-sonnet-4.6")

    with tempfile.TemporaryDirectory() as tmp:
        my_agent = CodeThenReview(workers.ClaudeCode(cwd=tmp))

        async with my_agent.run(
            model,
            [
                ai.user_message(
                    "Create hello.py that prints 'hello', then run it."
                )
            ],
        ) as stream:
            async for event in stream:
                if isinstance(event, ai.events.BuiltinToolEnd):
                    print(f"[worker] {event.tool_call.tool_name}")
                elif isinstance(event, ai.events.TextDelta):
                    print(event.chunk, end="", flush=True)
                elif isinstance(event, ai.events.StreamEnd):
                    print()

        print(f"review: {stream.output}")


if __name__ == "__main__":
    asyncio.run(main())
