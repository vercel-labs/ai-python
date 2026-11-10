"""A coding-agent CLI as a tool of an agent: the worker is a sub-agent."""

import asyncio
import tempfile

import ai
from ai import workers

model = ai.get_model("anthropic/claude-sonnet-4.6")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        worker = workers.ClaudeCode(cwd=tmp)

        # One session for the whole run: each call of the tool continues the
        # same conversation, on one live CLI.
        async with worker.session() as session:
            # The worker's events stream up as partial tool results, and its
            # last text becomes the tool result. To close the parent run
            # stops the worker's turn too.
            @ai.tool
            async def code(task: str) -> ai.SubAgentTool:
                """Do a coding task in the project; returns what was done."""
                messages = [*session.messages, ai.user_message(task)]
                async with ai.stream(worker, messages) as stream:
                    async for event in stream:
                        yield event

            orchestrator = ai.Agent(tools=[code])

            messages = [
                ai.system_message(
                    "You plan; the code tool writes the code. Summarize the "
                    "result for the user in one sentence."
                ),
                ai.user_message(
                    "Create hello.py that prints 'hello', then run it."
                ),
            ]

            async with orchestrator.run(model, messages) as stream:
                async for event in stream:
                    if isinstance(event, ai.events.PartialToolCallResult):
                        # The worker at work, inside the tool call.
                        inner = event.value
                        if isinstance(inner, ai.events.BuiltinToolEnd):
                            print(f"  [worker] {inner.tool_call.tool_name}")
                    elif isinstance(event, ai.events.TextDelta):
                        print(event.chunk, end="", flush=True)
            print()

            # The session has the worker's own conversation.
            print(f"worker messages: {len(session.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
