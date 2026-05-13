"""Human-in-the-loop approval hooks (inline gating variant).

Demonstrates the function-based hook API:
  - await hook("label", payload=Model) to suspend inside the loop
  - resolve_hook("label", data) to unblock from outside
  - Hook signals arrive as HookEvent events

The custom loop uses the concurrent ``ToolRunner`` flow.  The approval
hook is awaited inline inside the merge loop: on grant, the original
``ToolCall`` is scheduled normally; on denial, a synthetic
``ToolCallResult`` is fed straight into the runner via
``ToolRunner.add_result``.

This version uses a fully inline approach, with no external wrapper
class.

"""

import asyncio
from collections.abc import AsyncGenerator

import pydantic

import ai


class Approval(pydantic.BaseModel):
    granted: bool
    reason: str = ""


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


class ApprovalAgent(ai.Agent):
    TOOLS = [contact_mothership]

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        while context.keep_running():
            async with (
                ai.stream(context=context) as s,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(s, tr.events()):
                    yield event
                    if not isinstance(event, ai.events.ToolEnd):
                        continue

                    tc = context.resolve(event.tool_call)
                    if tc.name == "contact_mothership":
                        approval = await ai.hook(
                            f"approve_{tc.id}",
                            payload=Approval,
                            metadata={"tool": tc.name, "kwargs": tc.kwargs},
                        )
                        if not approval.granted:
                            tr.add_result(
                                ai.tool_result(
                                    tool_call_id=tc.id,
                                    tool_name=tc.name,
                                    result=f"Rejected: {approval.reason}",
                                    is_error=True,
                                )
                            )
                            continue

                    tr.schedule(tc)

                context.add(s.message)
                context.add(tr.get_tool_message())


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4")

    my_agent = ApprovalAgent()

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
                    Approval(
                        granted=answer.strip().lower() in ("y", "yes"),
                        reason="operator decision",
                    ),
                )
    print()


if __name__ == "__main__":
    asyncio.run(main())
