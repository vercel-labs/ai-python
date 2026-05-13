# ai

> [!WARNING]
> This framework is **experimental**. It is not stable and is not guaranteed to be maintained in the future. For evaluation purposes only.

Python toolkit for building model-powered apps and agent loops.

## Install

```bash
uv add vercel-ai-sdk
```

```python
import ai
```

## Quick Start

```python
import asyncio

import ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72F in {city}"


async def main() -> None:
    model = ai.ai_gateway("anthropic/claude-sonnet-4")
    agent = ai.agent(tools=[get_weather])

    messages = [
        ai.system_message("You are a helpful weather assistant."),
        ai.user_message("What's the weather in Tokyo?"),
    ]

    async with agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.TextDelta):
                print(event.chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
```

## API Surface

### Models

```
ai.openai(model_id)             provider — callable, returns Model
ai.anthropic(model_id)          provider — callable, returns Model
ai.ai_gateway(model_id)         provider — callable, returns Model
provider.list()                 list available model IDs from the provider API
ai.stream(model, messages, ...) streaming generation (supports tools=, output_type=)
ai.generate(model, messages, p) non-streaming generation (ImageParams, VideoParams)
ai.check_connection(model)      verify credentials and model availability
ai.Client(base_url=, api_key=)  explicit client — pass to provider call: ai.openai("gpt-5.4", client=c)
```

### Agents

```
ai.agent(tools=[...])           agent with tool loop
ai.tool                         decorator: schema gen + validation + execution
ai.hook(name, payload=, ...)    suspension point; resolve with ai.resolve_hook(...)
ai.resolve_hook(name, value)    resolve a pending hook from outside the loop
ai.cancel_hook(name)            cancel a pending hook
ai.yield_from(...)              forward nested agent / streaming tool output
```

### Messages

```
ai.system_message  ai.user_message  ai.assistant_message  ai.tool_message
ai.tool_result     ai.tool_result_part  ai.file_part  ai.thinking
```

### Middleware

```
ai.agents.Middleware             base class — subclass and override wrap methods
agent.run(..., middleware=[m])   run-scoped, first = outermost
```

Five wrap surfaces: `wrap_agent_run`, `wrap_model`, `wrap_generate`, `wrap_tool`, `wrap_hook`.
Each receives a context dataclass and a `next` callable. Default implementations pass through.

### Integrations

```
ai.mcp.get_http_tools(url, ...) expose an MCP server as tools
ai.ai_sdk_ui                    AI SDK UI streaming adapter
```

## Custom Agent Loops

Override the default loop when you need approval gates, routing, or custom orchestration:

```python
@agent.loop
async def custom(context: ai.Context):
    while True:
        async with ai.stream(context=context) as s:
            async for event in s:
                yield event
            context.add(s.message)

            tool_calls = context.resolve(s.tool_calls)
            if not tool_calls:
                return

            results = [await tc() for tc in tool_calls]
            yield ai.tool_result(*results)
```

## Examples

Small focused samples live in `examples/samples/`. End-to-end demos:

- `examples/fastapi-vite/` -- FastAPI backend + Vite frontend with hook-based tool approval
- `examples/multiagent-textual/` -- Textual TUI with parallel agents and interactive hook resolution
- `examples/temporal-direct/` -- durable agent with a custom loop (every I/O call is a Temporal activity)
- `examples/temporal-middleware/` -- durable agent via middleware (default loop, I/O routed to Temporal activities)
- `examples/samples/middleware_simple.py` -- middleware that logs all five execution surfaces
