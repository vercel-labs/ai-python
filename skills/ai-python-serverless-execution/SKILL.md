---
name: ai-python-serverless-execution
description: Use for Python ai SDK hook interruptions, approvals, replay, and resume across stateless requests.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-serverless-execution

Use hooks to stop a run, save messages, then replay from the same assistant
turn.

For tool approval, mark the tool:

```python
@ai.tool(require_approval=True)
async def delete_file(path: str) -> str:
    return f"Deleted {path}"
```

First request:

```python
async with agent.run(model, messages) as stream:
    async for event in stream:
        if (
            isinstance(event, ai.events.HookEvent)
            and event.hook.status == "pending"
        ):
            save_hook_id(event.hook.hook_id)
            ai.abort_pending_hook(event.hook)
        yield event

saved_messages = stream.messages
```

Next request:

```python
messages = [
    ai.messages.Message.model_validate(m)
    for m in load_messages_json()
]

ai.resolve_hook(
    hook_id,
    ai.tools.ToolApproval(granted=True, reason="approved"),
)

async with agent.run(model, messages) as stream:
    async for event in stream:
        yield event
```

Do not ask the model to make the tool call again. `Agent.run` prepares replay:
it marks the interrupted assistant turn with `replay=True` and keeps completed
sibling tool results on `cached_result`.

Persist messages, not the hook registry.
