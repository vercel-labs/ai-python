---
name: durable_execution
description: Use for Python ai SDK durable execution, serialization boundaries, replay, and workflow engines such as Temporal.
---

# durable_execution

Durability belongs at I/O boundaries. Keep the framework pieces when possible:
`Agent`, `Context`, messages, `ai.stream`, `ToolRunner`, and `@ai.tool`.

Serialize only data:

- Messages with `message.model_dump(mode="json")`.
- Restore with `ai.messages.Message.model_validate(...)`.
- Model IDs, tool schemas, params, and plain tool arguments.

Do not serialize live clients, provider instances with open clients, streams,
tasks, or sockets. Recreate those inside activities or other durable I/O calls.

For a durable model call, run the real model call in the durability API and
return a complete assistant `Message`. Feed it back through the normal stream
machinery:

```python
message = ai.messages.Message.model_validate(saved_message)

async with (
    ai.Stream(ai.events.replay_message_events(message)) as stream,
    ai.ToolRunner() as runner,
):
    async for event in ai.util.merge(stream, runner.events()):
        yield event
        if isinstance(event, ai.events.ToolEnd):
            runner.schedule(context.resolve(event.tool_call))

    context.add(stream.message)
    context.add(runner.get_tool_message())
```

For durable tools, pass `ToolRunner.schedule(...)` a zero-arg async callable
that calls the durability API and returns `ai.tool_result(...)`.

You do not need to rewrite the whole loop. Usually only the model call and tool
body need durable wrappers.
