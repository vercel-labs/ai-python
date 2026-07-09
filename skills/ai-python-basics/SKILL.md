---
name: ai-python-basics
description: Use for AI SDK for Python basics. Configure a model, make messages, stream, declare tools, build a basic agent.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-basics

Requires Python 3.12 or later. Install with `uv add ai`.

Use `import ai`.

For gateway-routed model IDs, set `AI_GATEWAY_API_KEY`.

Core pieces:

- `Model` selects the provider and model.
- Messages are typed Python objects.
- `ai.stream` makes one model call and returns one assistant message.
- `ai.Agent` wraps `ai.stream` in a loop that executes Python tools and manages
  history.

Use gateway model IDs unless you need a direct provider:

```python
model = ai.get_model("anthropic/claude-sonnet-4")
```

Direct providers need extras:

```bash
uv add "ai[openai]"
uv add "ai[anthropic]"
```

Messages are typed Python objects:

```python
messages = [
    ai.system_message("Be concise."),
    ai.user_message("Write a haiku about rain."),
]
```

Minimal agent happy path:

```python
import asyncio

import ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "Sunny"


async def main() -> None:
    model = ai.get_model("anthropic/claude-sonnet-4")
    agent = ai.Agent(tools=[get_weather])
    messages = [
        ai.system_message("Use tools when useful."),
        ai.user_message("What is the weather in San Francisco?"),
    ]

    async with agent.run(model, messages) as run:
        async for event in run:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)

    answer = run.output
    history = run.messages


if __name__ == "__main__":
    asyncio.run(main())
```

For one model call without Python tool execution, use `ai.stream`:

```python
async with ai.stream(model, messages) as stream:
    async for event in stream:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)

answer = stream.output
messages.append(stream.message)
```

Use `ai-python-custom-loop`, `ai-python-subagents`,
`ai-python-streaming-tools`, `ai-python-serverless-execution`,
`ai-python-durable-execution`, `ai-python-ui-adapter`, and
`ai-python-custom-provider` for advanced patterns.
