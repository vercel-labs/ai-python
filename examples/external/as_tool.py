"""A coding-agent CLI as a tool: the external harness is a sub-agent."""

import asyncio
import tempfile

import ai

model = ai.get_model("anthropic/claude-sonnet-4.6")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # One block for the whole run: each call of the tool continues the
        # same conversation, on one live CLI.
        async with ai.external.ClaudeCode(cwd=tmp) as harness:
            # The external harness's events stream up as partial tool results,
            # and its last text becomes the tool result. To close the parent run
            # stops the external harness's turn too.
            @ai.tool
            async def code(task: str) -> ai.SubAgentTool:
                """Do a coding task in the project; returns what was done."""
                messages = [
                    *harness.session.messages,
                    ai.user_message(task),
                ]
                async with ai.stream(harness, messages) as stream:
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
                    match event:
                        case ai.events.PartialToolCallResult(
                            value=ai.events.BuiltinToolEnd(tool_call=call)
                        ):
                            # The external harness at work, inside the tool
                            # call.
                            print(f"  [external harness] {call.tool_name}")
                        case ai.events.TextDelta():
                            print(event.chunk, end="", flush=True)
            print()

            # The session has the external harness's own conversation.
            count = len(harness.session.messages)
            print(f"external harness messages: {count}")


if __name__ == "__main__":
    asyncio.run(main())
