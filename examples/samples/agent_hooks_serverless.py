"""Serverless hook pattern.

Demonstrates the serverless/stateless pattern where the agent run suspends
cleanly when a hook has no resolution, then re-enters with a pre-registered
resolution.

Flow:
  1. First run: hook fires; the consumer sees the pending HookEvent and
     calls abort_pending_hook(), which raises HookPendingError at the
     awaiter so the gated tool short-circuits to a pending placeholder.
  2. Second run: resolve_hook() pre-registers the answer, agent.run()
     replays from the same input, and hook finds the resolution immediately.
"""

import asyncio

import ai

FILES_DELETED: set[str] = set()


@ai.tool(require_approval=True)
async def delete_file(path: str) -> str:
    """Delete a file at the given path."""
    print("FILE DELETED:", path)
    FILES_DELETED.add(path)
    return f"Deleted {path}"


AUDIT_LOG: list[str] = []


@ai.tool
async def audit_log(message: str) -> str:
    """Record a message in the audit log."""
    print("AUDIT LOG:", message)
    AUDIT_LOG.append(message)
    return f"Logged {message!r}"


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4.6")

    my_agent = ai.Agent(tools=[delete_file, audit_log])

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
                ai.abort_pending_hook(hook_part)

        # Pick up the assistant turn that the loop appended so the
        # next run replays from the same point.
        messages = stream.messages

    print(
        "\n  Run interrupted; approval will be pre-registered for re-entry.\n"
    )
    assert (
        len(AUDIT_LOG) == 1 and "/tmp/old_logs.txt" in AUDIT_LOG[0]
    ), f"Bad audit log: {AUDIT_LOG}"

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

    assert {
        "/tmp/old_logs.txt"
    } == FILES_DELETED, f"Wrong files deleted: {FILES_DELETED}"
    assert (
        len(AUDIT_LOG) == 1 and "/tmp/old_logs.txt" in AUDIT_LOG[0]
    ), f"Bad audit log: {AUDIT_LOG}"


if __name__ == "__main__":
    asyncio.run(main())
