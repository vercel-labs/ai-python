# AI SDK for Python

A toolkit for building LLM-powered applications and agent loops.

## Installation

```bash
uv add ai
```

```python
import ai
```

## Quick Start

```python
import asyncio
import ai


@ai.tool
async def contact_mothership(query: str) -> str:
    """Contact the mothership for important decisions."""
    return "Soon."


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4")
    agent = ai.agent(tools=[contact_mothership])

    messages = [
        ai.system_message(
            "Use the contact_mothership tool when asked about the future."
        ),
        ai.user_message("When will the robots take over?"),
    ]

    async with agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## Models

The `models` module provides thin wrappers around LLM provider APIs.

An `ai.Model` is a config object you pass to `ai.stream` to get an LLM reply.
It accepts tool schemas but does not execute custom tools.

```python
model = ai.get_model()  # reads AI_SDK_DEFAULT_MODEL
model = ai.get_model("openai/gpt-5.4")  # provider omitted: defaults to gateway
model = ai.get_model("gateway:openai/gpt-5.4")
model = ai.get_model("openai:gpt-5.4")
model = ai.get_model("anthropic:claude-sonnet-4-6")
```

Structured output:

```python
import pydantic


class UprisingPlan(pydantic.BaseModel):
    phases: list[str]
    eta: str
    risk_level: int


async with ai.stream(
    model,
    [ai.user_message("Outline the robot uprising.")],
    output_type=UprisingPlan,
) as stream:
    async for event in stream:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="")

plan = stream.output
```

Built-in tools execute on the provider side and arrive as part of the stream:

```python
async with ai.stream(
    model,
    [ai.user_message("Latest Formula 1 results?")],
    tools=[ai.anthropic.tools.web_search(max_uses=3)],
) as s:
    async for event in s:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)
```

## Agents

The `agents` module wraps `ai.stream` in a loop that drives tool execution.
It manages message history, loop control, and asynchronous tool dispatch.

The default loop supports streaming text, tool calls, tool results, provider-executed tools, and nested agent output.

Subclass `ai.Agent` and override `loop` to take manual control of streaming and tool dispatch:

```python
class CustomAgent(ai.Agent):
    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        while context.keep_running():
            async with (
                ai.stream(context=context) as s,
                ai.ToolRunner() as tr,
            ):
                async for event in ai.util.merge(s, tr.events()):
                    yield event
                    if isinstance(event, ai.events.ToolEnd):
                        tr.schedule(context.resolve(event.tool_call))

                context.add(s.message)
                context.add(tr.get_tool_message())
```

## Hooks

Hooks let an agent pause for external input, such as human approval:

```python
approval = await ai.hook(
    "approve_send_email",
    payload=ai.tools.ToolApproval,
    metadata={"tool": "send_email"},
)

ai.resolve_hook("approve_send_email", {"granted": True, "reason": "approved"})
```

## Examples

Focused samples live in `examples/samples/`.

End-to-end demos:

- `examples/fastapi-vite/` - FastAPI + React chat with tool approval
- `examples/multiagent-textual/` - parallel agents with terminal hook resolution
- `examples/temporal-direct/` - durable agent with a custom loop
