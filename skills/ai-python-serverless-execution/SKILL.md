---
name: ai-python-serverless-execution
description: Use when building serverless AI SDK for Python endpoints, handling hook approvals, deferring hooks, or resuming runs across requests.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-serverless-execution

Use this when working in a serverless setup, e.g. Vercel Fluid Compute.

The only major difference in serverless is processing tool approvals
and other hooks. Since you can't keep the hook future alive, you need
to stop the run, save messages, then start a later request with the
hook resolution pre-registered.

## Tool Approval

Mark approval-gated tools with `require_approval=True`:

```python
@ai.tool(require_approval=True)
async def delete_file(path: str) -> str:
    return f"Deleted {path}"
```

## First Request

When a deferred hook appears, send it to the client and call
`ai.defer_hook(...)`.

Keep draining the stream. Do not break after the first hook. This lets sibling
tools finish or get marked deferred, and makes `stream.messages` complete.

```python
deferred_hooks = []

async with agent.run(model, messages) as stream:
    async for event in stream:
        if (
            isinstance(event, ai.events.HookEvent)
            and event.hook.status == "pending"
        ):
            deferred_hooks.append(event.hook)
            ai.defer_hook(event.hook)

        yield event

saved_messages = [
    message.model_dump(mode="json")
    for message in stream.messages
]
save_messages(saved_messages)
save_deferred_hook_ids([hook.hook_id for hook in deferred_hooks])
```

## Resume Request

Load the saved messages, pre-register hook resolutions, then call `agent.run`.

```python
messages = [
    ai.messages.Message.model_validate(message)
    for message in load_messages()
]

for approval in approvals:
    ai.resolve_hook(
        approval.hook_id,
        ai.tools.ToolApproval(
            granted=approval.granted,
            reason=approval.reason,
        ),
    )

async with agent.run(model, messages) as stream:
    async for event in stream:
        yield event

save_messages([
    message.model_dump(mode="json")
    for message in stream.messages
])
```

Call `ai.resolve_hook(...)` before `agent.run(...)`. Do not ask the model to
make the tool call again.

`Agent.run` prepares saved interrupted messages for replay. Completed sibling
tool results are reused, deferred hooks receive the pre-registered resolution,
and replay-only events are hidden from the caller.

## Rules

- Use normal `agent.run(...)`; serverless resume usually does not need a custom loop.
- If you do write a custom loop, use `context.resolve(...)`, `ToolRunner`, and
  `context.add(...)` so approvals and replay keep working.
- For custom hooks, pre-register with `ai.resolve_hook(hook_id, data, payload=PayloadType)`.
- For AI SDK UI clients, use `ai-python-ui-adapter` for message conversion,
  approval responses, and SSE.
