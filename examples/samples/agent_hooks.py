"""Human-in-the-loop approval hooks.

Demonstrates the function-based hook API:
  - await hook("label", payload=Model) to suspend inside the loop
  - resolve_hook("label", data) to unblock from outside
  - Hook messages arrive with role="signal"
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


async def main() -> None:
    model = ai.model("ai-gateway", "anthropic/claude-sonnet-4")

    my_agent = ai.agent(tools=[contact_mothership])

    @my_agent.loop
    async def with_approval(context: ai.Context) -> AsyncGenerator[ai.Message]:
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
                if tc.name == "contact_mothership":
                    # Suspends until resolved from outside the loop.
                    approval = await ai.hook(
                        f"approve_{tc.id}",
                        payload=Approval,
                        metadata={"tool": tc.name, "kwargs": tc.kwargs},
                    )
                    if approval.granted:
                        results.append(await tc())
                    else:
                        results.append(
                            ai.Message(
                                role="tool",
                                parts=[
                                    ai.ToolResultPart(
                                        tool_call_id=tc.id,
                                        tool_name=tc.name,
                                        result=f"Rejected: {approval.reason}",
                                        is_error=True,
                                    )
                                ],
                            )
                        )
                else:
                    results.append(await tc())

            yield ai.tool_message(*results)

    messages = [
        ai.system_message(
            "Use the contact_mothership tool when asked about the future."
        ),
        ai.user_message("When will the robots take over?"),
    ]

    async for msg in my_agent.run(model, messages):
        # Hook signals arrive with role="signal"
        if msg.role == "signal":
            hook_part = msg.get_hook_part()
            if hook_part and hook_part.status == "pending":
                answer = input(f"Approve {hook_part.hook_id}? [y/n] ")
                ai.resolve_hook(
                    hook_part.hook_id,
                    Approval(
                        granted=answer.strip().lower() in ("y", "yes"),
                        reason="operator decision",
                    ),
                )
            continue

        if msg.text_delta:
            print(msg.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
