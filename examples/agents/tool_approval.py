"""Built-in tool approval: live resolution and interrupted replay."""

import asyncio
from typing import Any

import ai

model = ai.get_model("anthropic/claude-sonnet-4.6")

reports: list[str] = []


@ai.tool(require_approval=True)
async def notify(recipient: str, topic: str) -> str:
    """Send a short status report to a recipient."""
    reports.append(f"{recipient}:{topic}")
    return f"Sent status report about {topic} to {recipient}."


agent = ai.Agent(tools=[notify])


async def live(messages: list[ai.messages.Message]) -> None:
    print("--- live resolution ---")
    async with agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            elif (
                isinstance(event, ai.events.HookEvent)
                and event.hook.status == "pending"
            ):
                print(f"\n[pending] {event.hook.hook_id}")
                ai.resolve_hook(
                    event.hook,
                    ai.tools.ToolApproval(
                        granted=True,
                        reason="approved while the run is still active",
                    ),
                )
    print()


async def stateless(
    history: list[ai.messages.Message],
    approvals: dict[str, ai.tools.ToolApproval] | None = None,
) -> tuple[list[ai.messages.Message], list[ai.messages.HookPart[Any]], str]:
    for hook_id, approval in (approvals or {}).items():
        ai.resolve_hook(hook_id, approval)

    pending: list[ai.messages.HookPart[Any]] = []
    text: list[str] = []

    async with agent.run(model, history) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                text.append(event.chunk)
                print(event.chunk, end="", flush=True)
            elif (
                isinstance(event, ai.events.HookEvent)
                and event.hook.status == "pending"
            ):
                pending.append(event.hook)
                print(f"\n[pending] {event.hook.hook_id}")
                ai.abort_pending_hook(event.hook)

    return stream.messages, pending, "".join(text)


async def main() -> None:
    messages = [
        ai.system_message(
            "Always use notify before answering. "
            "Use recipient='ops' and the user's topic."
        ),
        ai.user_message(f"Send a status report about the database readiness."),
    ]

    await live(messages)

    print("\n--- interrupted resolution ---")
    messages, pending, _ = await stateless(messages)
    approvals = {
        hook.hook_id: ai.tools.ToolApproval(
            granted=True,
            reason="approved before replaying the interrupted run",
        )
        for hook in pending
    }
    print("\n\n--- replay with approval ---")
    await stateless(messages, approvals=approvals)

    print(f"\nReports sent: {', '.join(reports)}")


if __name__ == "__main__":
    asyncio.run(main())
