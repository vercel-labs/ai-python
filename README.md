# AI SDK for Python

A toolkit for building LLM-powered applications and agent loops.

> [!NOTE]
> The AI SDK for Python is in public beta.

## Installation

```bash
uv add ai
```

AI Gateway API-key usage works with the base package. Direct providers that
use an OpenAI-compatible, Anthropic-compatible, or Google adapter load the
corresponding official SDK lazily. Vercel OIDC for AI Gateway also uses an
optional extra:

```bash
uv add "ai[openai]"      # OpenAI-compatible providers
uv add "ai[anthropic]"   # Anthropic-compatible providers
uv add "ai[google]"      # Google Gemini
uv add "ai[vercel]"      # Vercel OIDC for AI Gateway
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
    model = ai.get_model("anthropic/claude-sonnet-4")
    agent = ai.Agent(tools=[contact_mothership])

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
model = ai.get_model("google:gemini-2.5-flash")
```

Provider IDs without a `provider:` prefix route through AI Gateway by default.
Direct OpenAI-compatible providers, including `openai:` and compatible
models.dev provider IDs, require `ai[openai]`. Direct Anthropic-compatible
providers require `ai[anthropic]`. The direct Google provider requires
`ai[google]`.

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
    tools=[ai.providers.anthropic.tools.web_search(max_uses=3)],
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

Focused samples live in category directories under `examples/`.

- `examples/agents/` - agent loops, tools, hooks, and MCP
- `examples/media/` - image, video, and multimodal input/output
- `examples/models/` - streaming, structured output, and provider examples
- `examples/apps/` - end-to-end demos

End-to-end demos:

- `examples/apps/web_agent/` - FastAPI + React chat with tool approval
- `examples/apps/coding_agent/` - coding agent
- `examples/apps/durable_agent_temporal/` - durable agent with Temporal
- `examples/apps/durable_agent_workflows/` - durable agent with Workflows
- `examples/apps/slack_agent/` - Slack agent
