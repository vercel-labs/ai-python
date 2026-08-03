---
name: ai
description: AI SDK for Python (the `ai` package). Use when writing Python that calls LLMs or builds agents — model calls, streaming, tool calling, subagents, human-in-the-loop approvals, durable/serverless execution, telemetry, AI SDK UI chat backends, custom providers.
metadata:
  sdk-version: "0.4.0"
---

# AI SDK for Python

Python SDK for calling LLMs and building agents. Package name: `ai`.
Requires Python 3.12+. Install with `uv add ai`. Use `import ai`.

Model IDs route through AI Gateway by default (set `AI_GATEWAY_API_KEY`).
Direct providers need extras and their own keys:

```bash
uv add "ai[openai]"     # OPENAI_API_KEY,    ai.get_model("openai:gpt-5")
uv add "ai[anthropic]"  # ANTHROPIC_API_KEY, ai.get_model("anthropic:claude-sonnet-4")
```

Core pieces:

- `Model` selects the provider and model: `ai.get_model("anthropic/claude-sonnet-4")`.
- Messages are typed Python objects: `ai.system_message(...)`, `ai.user_message(...)`.
- `ai.stream` makes one model call and returns one assistant message.
- `ai.Agent` wraps `ai.stream` in a loop that executes Python tools and manages history.

Minimal agent:

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

## Docs

Full docs are fetchable as plain markdown: take a docs path and append `.md`,
e.g. `https://ai-python.dev/docs/basics/tools.md`. Fetch the page for the task
at hand before writing code. Where a "local notes" file is listed below, read
it too — it contains rules the docs do not state.

| Task | Docs page | Local notes |
|---|---|---|
| Getting started | `https://ai-python.dev/docs/index.md` | this file |
| Models, gateway vs direct providers | `https://ai-python.dev/docs/basics/providers.md` | — |
| Single model call, structured output | `https://ai-python.dev/docs/basics/streaming.md` | — |
| Messages, events, serialization | `https://ai-python.dev/docs/basics/messages-and-events.md` | — |
| Declaring tools | `https://ai-python.dev/docs/basics/tools.md` | — |
| Streaming tool output, aggregators | `https://ai-python.dev/docs/basics/tools.md` | [references/streaming-tools.md](references/streaming-tools.md) |
| Agents, agent runs | `https://ai-python.dev/docs/basics/agents.md` | — |
| Subagents, multi-agent | `https://ai-python.dev/docs/basics/subagents-and-multi-agent.md` | [references/streaming-tools.md](references/streaming-tools.md) |
| Custom agent loops | `https://ai-python.dev/docs/basics/custom-loops.md` | [references/custom-loops.md](references/custom-loops.md) |
| Tool approvals, hooks, human-in-the-loop | `https://ai-python.dev/docs/basics/human-in-the-loop.md` | [references/serverless.md](references/serverless.md) |
| Serverless endpoints, resume across requests | `https://ai-python.dev/docs/basics/human-in-the-loop.md` | [references/serverless.md](references/serverless.md) |
| Durable execution / workflows | `https://ai-python.dev/docs/basics/durable-execution.md` | [references/durable.md](references/durable.md) |
| Telemetry, tracing, OpenTelemetry | `https://ai-python.dev/docs/basics/telemetry.md` | — |
| AI SDK UI (`useChat`) backends | `https://ai-python.dev/docs/basics/ai-sdk-ui.md` | [references/ui.md](references/ui.md) |
| Writing a custom provider | `https://ai-python.dev/docs/basics/providers.md` | [references/custom-provider.md](references/custom-provider.md) |

Reference pages live under `https://ai-python.dev/docs/reference.md`
(`reference/ai.md`, `reference/messages.md`, `reference/events.md`,
`reference/tools.md`, `reference/telemetry.md`, `reference/errors.md`,
`reference/types.md`, `reference/util.md`, `reference/mcp.md`).
