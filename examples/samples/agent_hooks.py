"""Human-in-the-loop approval hooks.

Demonstrates the function-based hook API:
  - await hook("label", payload=Model) to suspend inside the loop
  - resolve_hook("label", data) to unblock from outside
  - Hook signals arrive as HookEvent events

The custom loop uses the concurrent ``ToolRunner`` flow: tools are
scheduled and run concurrently as the model emits them.  The approval
hook is awaited inside a ``ToolCall``-shaped wrapper that is scheduled
in place of the bare tool call, so gating composes naturally with the
runner's merge-and-iterate behaviour.
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


class GatedCall:
    """ToolCall-shaped wrapper that awaits an approval hook before executing.

    ``ToolRunner.schedule`` only consumes the ``__call__`` shape of
    ``ToolCall``; this wrapper supplies the same shape while inserting
    the hook await + denial path before the underlying tool runs.
    """

    def __init__(self, tc: ai.ToolCall) -> None:
        self._tc = tc

    async def __call__(self) -> ai.events.ToolCallResult:
        tc = self._tc
        approval = await ai.hook(
            f"approve_{tc.id}",
            payload=Approval,
            metadata={"tool": tc.name, "kwargs": tc.kwargs},
        )
        if approval.granted:
            return await tc()
        return ai.tool_result(
            tool_call_id=tc.id,
            tool_name=tc.name,
            result=f"Rejected: {approval.reason}",
            is_error=True,
        )


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[contact_mothership])

    @my_agent.loop
    async def with_approval(
        context: ai.Context,
    ) -> AsyncGenerator[ai.events.AgentEvent]:
        while context.keep_running():
            async with (
                ai.stream(context.model, context.messages, tools=context.tools) as s,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(s, tr.events()):
                    yield event
                    if isinstance(event, ai.events.ToolEnd):
                        tc = context.resolve(event.tool_call)
                        if tc.name == "contact_mothership":
                            tr.schedule(GatedCall(tc))
                        else:
                            tr.schedule(tc)

                context.add(s.message)
                context.add(tr.get_tool_message())

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
