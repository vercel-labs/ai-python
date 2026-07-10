"""Custom hook: pause a tool for application-specific input."""

import asyncio

import pydantic

import ai


class DeploymentReview(pydantic.BaseModel):
    approved: bool
    reviewer: str
    reason: str = ""


@ai.tool
async def request_deployment_review(environment: str, change: str) -> str:
    """Request external review before deployment."""
    review = await ai.hook(
        f"deployment_review_{environment}",
        payload=DeploymentReview,
        metadata={"environment": environment, "change": change},
    )
    if review.approved:
        return f"Approved by {review.reviewer}: {review.reason}"
    return f"Rejected by {review.reviewer}: {review.reason}"


async def main() -> None:
    model = ai.get_model("anthropic/claude-sonnet-4.6")
    agent = ai.Agent(tools=[request_deployment_review])

    messages = [
        ai.system_message(
            "Always call request_deployment_review before answering "
            "deployment questions."
        ),
        ai.user_message(
            "Can I deploy version 2.7 to production during the next window?"
        ),
    ]

    async with agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            elif (
                isinstance(event, ai.events.HookEvent)
                and event.hook.status == "pending"
            ):
                print(f"\n[deferred] {event.hook.hook_id}")
                ai.resolve_hook(
                    event.hook,
                    DeploymentReview(
                        approved=True,
                        reviewer="ops",
                        reason="change window is open",
                    ),
                )
            elif (
                isinstance(event, ai.events.HookEvent)
                and event.hook.status == "resolved"
            ):
                print(f"[resolved] {event.hook.hook_id}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
