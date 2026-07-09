---
name: ai-python-streaming-tools
description: Use for AI SDK for Python async-generator tools, streaming tool output, subagent tools, PartialToolCallResult events, and custom tool aggregation.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-streaming-tools

Use async-generator tools when a tool should show progress while it runs.

A streaming tool yields many values, but the agent still needs one final tool
result for the next model turn. The return type tells the SDK how to combine
those yields.

## Text Chunks

Use `ai.StreamingTextTool` when yielded strings should concatenate into the
final tool result.

```python
@ai.tool
async def draft_reply(topic: str) -> ai.StreamingTextTool:
    yield "Checking records for "
    yield topic
```

The caller sees each yield as `ai.events.PartialToolCallResult`. The model later
sees one string: `"Checking records for {topic}"`.

## Progress Then Result

Use `ai.StreamingStatusTool[T]` when the last yielded value is the tool result.

```python
@ai.tool
async def ask_mothership(question: str) -> ai.StreamingStatusTool[str]:
    yield "connecting"
    yield "transmitting"
    yield f"The mothership says: {question} is under review."
```

The progress values stream to the caller. The model sees only the final yielded
value.

## Subagents

Use `ai.SubAgentTool` when a tool runs another agent and streams its events.

```python
@ai.tool
async def research(topic: str) -> ai.SubAgentTool:
    researcher = ai.Agent()

    messages = [
        ai.system_message("Research briefly."),
        ai.user_message(topic),
    ]

    async with researcher.run(model, messages) as stream:
        async for event in stream:
            yield event
```

The parent caller receives nested events as `PartialToolCallResult.value`.

```python
async with agent.run(model, messages) as stream:
    async for event in stream:
        if isinstance(event, ai.events.PartialToolCallResult):
            if isinstance(event.value, ai.events.TextDelta):
                yield event.value.chunk
```

`SubAgentTool` stores a typed `MessageBundle` as the tool result. The parent
model sees the nested agent's final assistant text, not the raw bundle.

When saving history, keep the typed message data:

```python
data = message.model_dump(mode="json")
message = ai.messages.Message.model_validate(data)
```

Do not stringify `MessageBundle` or drop `result_kind`.

## Custom Aggregation

Prefer the aliases above. If you need custom aggregation, use either
`@ai.tool(aggregator=...)` or an `Annotated` return type. Do not use both.

```python
from collections.abc import AsyncGenerator
from typing import Annotated

JoinedLines = Annotated[
    AsyncGenerator[str],
    ai.agents.Aggregate(ai.agents.ConcatAggregator, delim="\n"),
]


@ai.tool
async def outline(topic: str) -> JoinedLines:
    yield f"# {topic}"
    yield "- first point"
```

Custom aggregators implement `ai.events.Aggregator`.

## Rules

- Streaming tools must be async generators.
- Every streaming tool needs an aggregator, usually from the return type alias.
- Consume live output from `ai.events.PartialToolCallResult`.
- The final aggregated value is sent back to the model as a normal tool result.
- In custom agent loops, keep `ToolRunner` events flowing; otherwise partial
  tool output will not reach the caller.
