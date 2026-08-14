---
name: ai
description: AI SDK for Python (the `ai` package). Use when writing Python that calls LLMs or dedicated image, video, speech, embedding, transcription, or reranking models; builds agents; tests model interactions; or implements tool calling, subagents, approvals, durable execution, telemetry, AI SDK UI backends, and custom providers.
metadata:
  sdk-version: "0.5.0"
---

# AI SDK for Python

Package: `ai`. Requires Python 3.12+. Install with `uv add ai`.

Unprefixed model IDs use AI Gateway and `AI_GATEWAY_API_KEY`. Direct providers
use `provider:model`, their API key, and an extra:

```bash
uv add "ai[openai]"     # OPENAI_API_KEY,    ai.get_model("openai:gpt-5")
uv add "ai[anthropic]"  # ANTHROPIC_API_KEY, ai.get_model("anthropic:claude-sonnet-4")
```

## Basic use

Use `ai.stream` for one model call without Python tool execution:

```python
import ai

model = ai.get_model("anthropic/claude-sonnet-4")
messages = [
    ai.system_message("Be concise."),
    ai.user_message("Write a haiku about rain."),
]

async with ai.stream(model, messages) as stream:
    async for event in stream:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)

answer = stream.output
message = stream.message
```

Use `ai.Agent` for a loop that executes Python tools and manages history:

```python
@ai.tool
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "Sunny"


agent = ai.Agent(tools=[get_weather])
async with agent.run(model, messages) as run:
    async for event in run:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)

answer = run.output
history = run.messages
```

These examples are sufficient for basic model calls, messages, tools, and
agents.

## Advanced work

For an advanced task, fetch its page under `https://ai-python.dev/docs/` and
read the listed local notes before writing code.

| Task | Page | Local notes |
|---|---|---|
| Provider clients, options, discovery | `basics/providers.md` | — |
| Structured output, complex streams | `basics/streaming.md` | — |
| Buffered language-model calls | `basics/streaming.md` | — |
| Images, video, speech, embeddings, transcription, reranking | `basics/model-operations.md` | — |
| Events and serialization | `basics/messages-and-events.md` | — |
| Advanced tools, streaming, aggregation | `basics/tools.md` | [streaming-tools.md](references/streaming-tools.md) |
| Advanced agent behavior | `basics/agents.md` | — |
| Deterministic model and agent tests | `basics/testing.md` | — |
| Subagents and multi-agent | `basics/subagents-and-multi-agent.md` | [streaming-tools.md](references/streaming-tools.md) |
| Custom agent loops | `basics/custom-loops.md` | [custom-loops.md](references/custom-loops.md) |
| Approvals and hooks | `basics/human-in-the-loop.md` | — |
| Serverless resume | `basics/human-in-the-loop.md` | [serverless.md](references/serverless.md) |
| Durable execution | `basics/durable-execution.md` | [durable.md](references/durable.md) |
| Telemetry and tracing | `basics/telemetry.md` | — |
| AI SDK UI backends | `basics/ai-sdk-ui.md` | [ui.md](references/ui.md) |
| Custom providers | `basics/providers.md` | [custom-provider.md](references/custom-provider.md) |

For exact APIs, use `reference.md` and the relevant `reference/*.md` page.
