"""Make a harness turn one step of your own agent loop.

The harness does the coding; a model reviews the result. The loop is
ordinary `ai.Agent.loop` code: it chooses which harness events to pass on
(reasoning is dropped here), what joins the agent's context, and what runs
next — here a check in the workspace, then a model call.

The harness ran its own tools. Its `ToolEnd` events are reports, so this
loop never hands them to `self.resolve()` or a `ToolRunner`.
"""

import asyncio
import tempfile
from collections.abc import AsyncGenerator

import ai
from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces

REASONING = (
    ai.events.ReasoningStart,
    ai.events.ReasoningDelta,
    ai.events.ReasoningEnd,
)


class CodeThenReview(ai.Agent):
    def __init__(self, harness: harnesses.Harness) -> None:
        super().__init__()
        self.harness = harness

    async def loop(
        self, context: ai.Context
    ) -> AsyncGenerator[ai.events.AgentEvent]:
        # Step 1: the harness does the task.
        async with self.harness.session().stream(
            context.messages[-1].text
        ) as turn:
            async for event in turn:
                if not isinstance(event, REASONING):
                    yield event
        if turn.result.cancelled:
            return

        # Step 2: evidence from the workspace, not the harness's word for
        # it. Only the harness's answer joins the agent's context, not its
        # whole transcript: its tool calls name tools this model lacks.
        workspace = self.harness.workspace
        source = await workspace.read_text("hello.py")
        run = await workspace.exec(["python3", "hello.py"], timeout=30)
        context.add(ai.assistant_message(turn.output))
        context.add(
            ai.user_message(
                "You are the reviewer. hello.py contains:\n"
                + source
                + f"\nRunning it printed: {run.stdout.strip()!r} "
                + f"(exit code {run.exit_code}).\n"
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
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp)
        ) as harness:
            agent = CodeThenReview(harness)
            task = "Create hello.py that prints 'hello', then run it."
            async with agent.run(model, [ai.user_message(task)]) as stream:
                async for event in stream:
                    if isinstance(event, ai.events.ToolEnd):
                        print(f"[harness] {event.tool_call.tool_name}")
                    elif isinstance(event, ai.events.TextDelta):
                        print(event.chunk, end="", flush=True)
                    elif isinstance(event, ai.events.StreamEnd):
                        print()
            print(f"review: {stream.output}")


asyncio.run(main())
