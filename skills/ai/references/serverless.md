# Serverless resume: persistence pattern

Read `https://ai-python.dev/docs/basics/human-in-the-loop.md` first
("Resume in serverless flows"). This file adds the request/persistence
mechanics the docs do not show.

In serverless (e.g. Vercel Fluid Compute) you cannot keep a hook future alive
across requests. Stop the run, save messages, then start a later request with
the hook resolution pre-registered.

## First request

When a deferred hook appears, send it to the client and call
`ai.defer_hook(...)`. Keep draining the stream — do not break after the first
hook. This lets sibling tools finish or get marked deferred, and makes
`stream.messages` complete.

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

## Resume request

Load the saved messages, then pre-register hook resolutions with
`ai.resolve_hook(...)`. Pre-registration must target the run's registry:
call it inside the `async with agent.run(...)` block before iterating the
stream, or pass `registry=` explicitly.

```python
messages = [
    ai.messages.Message.model_validate(message)
    for message in load_messages()
]

async with agent.run(model, messages) as stream:
    for approval in approvals:
        ai.resolve_hook(
            approval.hook_id,
            ai.tools.ToolApproval(
                granted=approval.granted,
                reason=approval.reason,
            ),
        )

    async for event in stream:
        yield event

save_messages([
    message.model_dump(mode="json")
    for message in stream.messages
])
```

## Rules

- Do not ask the model to make the tool call again; replay reuses completed
  sibling results and feeds deferred hooks the pre-registered resolution.
- Use normal `agent.run(...)`; serverless resume usually does not need a
  custom loop. If you do write one, use `context.resolve(...)`, `ToolRunner`,
  and `context.add(...)` so approvals and replay keep working.
- For custom hooks, pre-register with
  `ai.resolve_hook(hook_id, data, payload=PayloadType)` to validate the data.
- For AI SDK UI clients, see [ui.md](ui.md) for message conversion, approval
  responses, and SSE.
