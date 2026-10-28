"""Use a coding-agent CLI as a tool of an `ai.Agent`.

A harness is an agent, so it fits where a sub-agent does: a plain tool that
yields the turn's events. The parent model delegates a whole coding task,
the harness's own steps stream up as `PartialToolCallResult`s, and the
parent model sees the harness's final answer as the tool result.

Closing the parent run stops the harness's turn too.
"""

import asyncio
import tempfile

import ai
from ai.harnesses import experimental as harnesses
from ai.workspaces import experimental as workspaces

model = ai.get_model("anthropic/claude-sonnet-4.6")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        async with harnesses.claude_code(
            workspace=workspaces.Local(tmp)
        ) as harness:
            session = harness.session()

            @ai.tool
            async def code(task: str) -> ai.SubAgentTool:
                """Do a coding task in the project; returns what was done."""
                async with session.stream(task) as turn:
                    async for event in turn:
                        yield event

            orchestrator = ai.Agent(tools=[code])
            messages = [
                ai.system_message(
                    "You plan; the code tool writes the code. Summarize the "
                    + "result for the user in one sentence."
                ),
                ai.user_message(
                    "Create hello.py that prints 'hello', then run it."
                ),
            ]
            async with orchestrator.run(model, messages) as stream:
                async for event in stream:
                    if isinstance(event, ai.events.PartialToolCallResult):
                        # The harness at work, inside the tool call.
                        inner = event.value
                        if isinstance(inner, ai.events.ToolEnd):
                            print(f"  [harness] {inner.tool_call.tool_name}")
                    elif isinstance(event, ai.events.TextDelta):
                        print(event.chunk, end="", flush=True)
            print()
            # The harness kept its own transcript of what it did.
            print(f"harness messages: {len(session.messages)}")


asyncio.run(main())
