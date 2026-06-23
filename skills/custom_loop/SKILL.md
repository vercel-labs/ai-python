---
name: custom_loop
description: Use when changing a Python ai SDK agent loop for custom tool order, routing, logging, hooks, replay, or scheduling.
---

# custom_loop

Keep the default shape unless you must change control flow:

```python
class MyAgent(ai.Agent):
    async def loop(self, context: ai.Context):
        while context.keep_running():
            async with (
                ai.stream(context=context) as stream,
                ai.ToolRunner() as runner,
            ):
                async for event in ai.util.merge(stream, runner.events()):
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        runner.schedule(context.resolve(event.tool_call))

                context.add(stream.message)
                context.add(runner.get_tool_message())
```

Rules:

- Call `context.keep_running()` at the top of each turn.
- Use `ai.stream(context=context)` so model, messages, tools, output type, and params stay together.
- Yield events from the loop. `Agent.run` hides replay events from callers.
- On `ToolEnd`, use `context.resolve(event.tool_call)`. It handles validation, approval gates, and cached replay results.
- Do not call `tool.fn` directly unless you also handle validation, approvals, and cached results.
- Schedule resolved calls with `ToolRunner.schedule(...)`.
- `ToolRunner.schedule(...)` also accepts a zero-arg async callable that returns `ai.events.ToolCallResult`.
- If you make a result yourself, use `runner.add_result(ai.tool_result(...))`.
- Add `stream.message`, then `runner.get_tool_message()`. `context.add(...)` skips replay messages.
- Every tool call must get one tool result.
- For hooks, let `context.resolve(...)` build the gated call. Use `serverless_execution` for request boundaries.
- For durable calls, keep this shape and wrap only model or tool I/O. Use `durable_execution`.
