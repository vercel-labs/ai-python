---
name: ai-python-subagents
description: Use for the subagent-as-a-tool pattern.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-subagents

Use a subagent tool when the parent model should choose when to call another
agent.

```python
@ai.tool
async def research(topic: str) -> ai.SubAgentTool:
    child = ai.Agent(tools=[lookup])
    messages = [
        ai.system_message("Research briefly."),
        ai.user_message(topic),
    ]

    async with child.run(model, messages) as stream:
        async for event in stream:
            yield event
```

The parent model sees the child agent's final assistant text as the tool
result. The caller sees child events as `ai.events.PartialToolCallResult`.

Render nested output from `event.value`:

```python
if isinstance(event, ai.events.PartialToolCallResult):
    if isinstance(event.value, ai.events.TextDelta):
        print(event.value.chunk, end="", flush=True)
```

Do not append child messages to the parent history yourself. The tool result
stores the child transcript as a `MessageBundle`.

For `MessageBundle`, preliminary output, and streaming tool details, use
`ai-python-streaming-tools`.
