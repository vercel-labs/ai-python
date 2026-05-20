"""Human-in-the-loop approval hooks.

Demonstrates the function-based hook API:
  - mark a tool with ``require_approval=True`` to gate its execution
    behind an approval hook
  - resolve_hook("label", data) to unblock from outside
  - Hook signals arrive as HookEvent events
"""

import asyncio

import ai


@ai.tool(require_approval=True)
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")

    my_agent = ai.Agent(tools=[contact_mothership])

    messages = [
        ai.system_message(
            "Use the contact_mothership tool when asked about the future."
        ),
        ai.user_message("When will the robots take over?"),
    ]

    async with my_agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
                continue

            # Hook signals arrive as HookEvent events.
            if (
                isinstance(event, ai.events.HookEvent)
                and event.hook.status == "pending"
            ):
                hook_part = event.hook
                answer = input(f"Approve {hook_part.hook_id}? [y/n] ")
                ai.resolve_hook(
                    hook_part.hook_id,
                    ai.tools.ToolApproval(
                        granted=answer.strip().lower() in ("y", "yes"),
                        reason="operator decision",
                    ),
                )
    print()


if __name__ == "__main__":
    asyncio.run(main())
