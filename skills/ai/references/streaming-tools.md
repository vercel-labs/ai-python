# Streaming tools and subagents: extra rules

Read `https://ai-python.dev/docs/basics/tools.md` (streaming tools) and
`https://ai-python.dev/docs/basics/subagents-and-multi-agent.md` first.
Additional rules:

## Custom aggregation

Prefer the built-in aliases (`ai.StreamingTextTool`, `ai.StreamingStatusTool[T]`,
`ai.SubAgentTool`). If you need custom aggregation, use either
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

- Streaming tools must be async generators, and every streaming tool needs an
  aggregator — usually supplied by the return type alias.
- Do not append a subagent's child messages to the parent history yourself.
  The tool result stores the child transcript as a typed `MessageBundle`.
- When saving history, keep the typed message data. Do not stringify
  `MessageBundle` or drop `result_kind`:

  ```python
  data = message.model_dump(mode="json")
  message = ai.messages.Message.model_validate(data)
  ```
