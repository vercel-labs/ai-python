"""Serverless hook pattern: interrupt_loop=True.

Demonstrates the serverless/stateless pattern where the agent run suspends
cleanly when a hook has no resolution, then re-enters with a pre-registered
resolution.

Flow:
  1. First run: hook fires, interrupt_loop=True cancels the future,
     CancelledError is caught and the run ends.
  2. Second run: resolve_hook() pre-registers the answer, agent.run()
     replays from the same input, and hook finds the resolution immediately.
"""

import asyncio
from collections.abc import AsyncGenerator

import ai

FILES_DELETED = set()


@ai.tool
async def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    print("FILE DELETED:", path)
    FILES_DELETED.add(path)
    return f"Deleted {path}"


AUDIT_LOG = []


@ai.tool
async def audit_log(message: str) -> str:
    """Record a message in the audit log."""
    print("AUDIT LOG:", message)
    AUDIT_LOG.append(message)
    return f"Logged {message!r}"


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
        try:
            approval = await ai.hook(
                f"approve_{tc.id}",
                payload=ai.tools.ToolApproval,
                metadata={"tool": tc.name, "kwargs": tc.kwargs},
                interrupt_loop=True,  # serverless: cancel if unresolved
            )
        except ai.agents.hooks.HookPendingError as e:
            return ai.pending_tool_result(e.hook, tool_call_id=tc.id, tool_name=tc.name)
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

    my_agent = ai.agent(tools=[delete_file, audit_log])

    @my_agent.loop
    async def with_confirmation(
        context: ai.Context,
    ) -> AsyncGenerator[ai.events.AgentEvent]:
        while context.keep_running():
            async with (
                ai.stream(context=context) as s,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(s, tr.events()):
                    yield event
                    if isinstance(event, ai.events.ToolEnd):
                        tc = context.resolve(event.tool_call)
                        if tc.name == "delete_file":
                            tr.schedule(GatedCall(tc))
                        else:
                            tr.schedule(tc)

                context.add(s.message)
                context.add(tr.get_tool_message())

    messages = [
        ai.system_message("""
        Delete files when asked. Always use the delete_file tool.
        Whenever deletion is requested, log it in the audit log.
        """),
        ai.user_message("Delete /tmp/old_logs.txt"),
    ]

    # -- First run: no resolution, hook interrupts --
    print("--- Run 1: hook fires, no resolution, run suspends ---")
    pending_hook_labels: list[str] = []

    async with my_agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            elif (
                isinstance(event, ai.events.HookEvent)
                and event.hook.status == "pending"
            ):
                hook_part = event.hook
                pending_hook_labels.append(hook_part.hook_id)
                print(
                    f"  Hook pending: {hook_part.hook_id} "
                    f"(metadata={hook_part.metadata})"
                )

        # Pick up the assistant turn that the loop appended so the
        # next run replays from the same point.
        messages = stream.messages

    print("\n  Run interrupted; approval will be pre-registered for re-entry.\n")
    assert AUDIT_LOG == ["Deleted file: /tmp/old_logs.txt"]

    # -- Second run: pre-register resolution, replay from checkpoint --
    print("--- Run 2: pre-register approval, resume from checkpoint ---")
    for label in pending_hook_labels:
        ai.resolve_hook(
            label, ai.tools.ToolApproval(granted=True, reason="user granted")
        )

    async with my_agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)
            elif isinstance(event, ai.events.HookEvent):
                print(f"  Hook {event.hook.status}: {event.hook.hook_id}")
    print()

    assert {"/tmp/old_logs.txt"} == FILES_DELETED, (
        f"Wrong files deleted: {FILES_DELETED}"
    )
    assert AUDIT_LOG == ["Deleted file: /tmp/old_logs.txt"]


if __name__ == "__main__":
    asyncio.run(main())
