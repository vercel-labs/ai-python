---
name: ai-python-streaming-tools
description: Use for Python ai SDK tools that stream partial output, nested agent events, MessageBundle values, or preliminary UI output.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-streaming-tools

Make a streaming tool an async generator.

Use `StreamingTextTool` when yielded strings join into the result:

```python
@ai.tool
async def draft(topic: str) -> ai.StreamingTextTool:
    yield "Drafting "
    yield topic
```

Use `StreamingStatusTool[T]` when only the last yield is the result:

```python
@ai.tool
async def fetch(url: str) -> ai.StreamingStatusTool[str]:
    yield "connecting"
    yield "downloading"
    yield body
```

Use `SubAgentTool` for nested agent events:

```python
@ai.tool
async def research(topic: str) -> ai.SubAgentTool:
    async with child.run(model, [ai.user_message(topic)]) as stream:
        async for event in stream:
            yield event
```

Live values arrive as `ai.events.PartialToolCallResult`.

`SubAgentTool` stores a `MessageBundle` as the tool result. The model sees the
last child assistant text, not the raw bundle. Keep the bundle typed when you
save history:

```python
data = message.model_dump(mode="json")
message = ai.messages.Message.model_validate(data)
```

Do not stringify `MessageBundle` or drop `result_kind`. The UI adapter uses it
to round-trip subagent output.

For custom aggregation, use `ai.agents.Aggregate` on the return type.
