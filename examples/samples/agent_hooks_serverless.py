"""Serverless hook pattern: interrupt_loop=True.

Demonstrates the serverless/stateless pattern where the agent run suspends
cleanly when a hook has no resolution, and resumes from a checkpoint on
re-entry with a pre-registered resolution.

Flow:
  1. First run: hook fires, interrupt_loop=True cancels the future,
     CancelledError is caught, run ends with a checkpoint.
  2. Second run: resolve_hook() pre-registers the answer, agent.run()
     replays from checkpoint, hook finds the resolution immediately.
"""

import asyncio
from collections.abc import AsyncGenerator

import pydantic

import ai


class Confirmation(pydantic.BaseModel):
    approved: bool
    reason: str = ""


@ai.tool
async def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    return f"Deleted {path}"


async def main() -> None:
    model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[delete_file])

    @my_agent.loop
    async def with_confirmation(context: ai.Context) -> AsyncGenerator[ai.Message]:
        while True:
            s = await ai.models.stream(
                context.model, context.messages, tools=context.tools
            )
            async for msg in s:
                yield msg

            tool_calls = context.resolve(s.tool_calls)
            if not tool_calls:
                return

            results = []
            for tc in tool_calls:
                try:
                    confirmation = await ai.hook(
                        f"confirm_{tc.id}",
                        payload=Confirmation,
                        metadata={"tool": tc.name, "kwargs": tc.kwargs},
                        interrupt_loop=True,  # serverless: cancel if unresolved
                    )
                except asyncio.CancelledError:
                    # No resolution available — bail out cleanly.
                    # The checkpoint captures everything up to this point.
                    return

                if confirmation.approved:
                    results.append(await tc())
                else:
                    results.append(
                        ai.Message(
                            role="tool",
                            parts=[
                                ai.ToolResultPart(
                                    tool_call_id=tc.id,
                                    tool_name=tc.name,
                                    result=f"Rejected: {confirmation.reason}",
                                    is_error=True,
                                )
                            ],
                        )
                    )

            yield ai.tool_message(*results)

    messages = [
        ai.system_message("Delete files when asked. Always use the delete_file tool."),
        ai.user_message("Delete /tmp/old_logs.txt"),
    ]

    # -- First run: no resolution, hook interrupts --
    print("--- Run 1: hook fires, no resolution, run suspends ---")
    pending_hook_labels: list[str] = []

    durability = ai.EventLogProvider()
    async for msg in my_agent.run(model, messages, durability=durability):
        if msg.role == "signal":
            hook_part = msg.get_hook_part()
            if hook_part and hook_part.status == "pending":
                pending_hook_labels.append(hook_part.hook_id)
                print(
                    f"  Hook pending: {hook_part.hook_id}"
                    f" (metadata={hook_part.metadata})"
                )
        elif msg.text_delta:
            print(msg.text_delta, end="", flush=True)

    saved_checkpoint = durability.checkpoint()
    print(f"\n  Checkpoint saved: {len(saved_checkpoint.steps)} steps\n")

    # -- Second run: pre-register resolution, replay from checkpoint --
    print("--- Run 2: pre-register approval, resume from checkpoint ---")
    for label in pending_hook_labels:
        ai.resolve_hook(label, Confirmation(approved=True, reason="user approved"))

    durability = ai.EventLogProvider(saved_checkpoint)
    async for msg in my_agent.run(model, messages, durability=durability):
        if msg.role == "signal":
            hook_part = msg.get_hook_part()
            if hook_part:
                print(f"  Hook {hook_part.status}: {hook_part.hook_id}")
        elif msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
