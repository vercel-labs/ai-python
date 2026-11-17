"""Custom loop: an external harness turn as one step of your own agent loop."""

import asyncio
import pathlib
import subprocess
import tempfile
from collections.abc import AsyncGenerator

import ai

REASONING = (
    ai.events.ReasoningStart,
    ai.events.ReasoningDelta,
    ai.events.ReasoningEnd,
)


class CodeThenReview(ai.Agent):
    def __init__(self, harness: ai.external.ExternalHarness) -> None:
        super().__init__()
        self.harness = harness

    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[ai.events.AgentEvent]:
        """The external harness does the task, then the model reviews it."""
        # Step 1: the external harness does the task. Each run is a new
        # conversation, thus a new session.
        harness = self.harness
        async with (
            harness.session(),
            ai.stream(harness, [context.messages[-1]]) as work,
        ):
            async for event in work:
                if not isinstance(event, REASONING):
                    yield event

        # Step 2: evidence from the directory, not the external harness's word
        # for it. Only the external harness's answer joins the agent's context,
        # not its tool calls: they name tools that this model does not have.
        source = pathlib.Path(harness.cwd, "hello.py").read_text()
        run = await asyncio.to_thread(
            lambda: subprocess.run(
                ["python3", "hello.py"],
                cwd=harness.cwd,
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
        my_agent = CodeThenReview(ai.external.ClaudeCode(cwd=tmp))

        async with my_agent.run(
            model,
            [
                ai.user_message(
                    "Create hello.py that prints 'hello', then run it."
                )
            ],
        ) as stream:
            async for event in stream:
                match event:
                    case ai.events.BuiltinToolEnd():
                        print(f"[external harness] {event.tool_call.tool_name}")
                    case ai.events.TextDelta():
                        print(event.chunk, end="", flush=True)
                    case ai.events.StreamEnd():
                        print()

        print(f"review: {stream.output}")


if __name__ == "__main__":
    asyncio.run(main())
