---
name: ai
description: Use for Python ai SDK basics: models, messages, streaming, tools, agents, and the minimal happy path.
---

# ai

Install with `uv add ai`.

Use `import ai`.

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

For one model call, use `ai.stream`:

```python
async with ai.stream(model, messages) as stream:
    async for event in stream:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)

answer = stream.output
messages.append(stream.message)
```

For Python tools, use an agent:

```python
@ai.tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "Sunny"

agent = ai.agent(tools=[get_weather])

async with agent.run(model, messages) as run:
    async for event in run:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)

answer = run.output
history = run.messages
```

Use `custom_loop`, `subagents`, `streaming_tools`, `serverless_execution`,
`durable_execution`, `ai_sdk_ui_adapter`, and `custom_provider` for advanced
patterns.
