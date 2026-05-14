---
name: ai
description: Python `ai` SDK — models, providers, streams, events, tools, agents, hooks, MCP, AI SDK UI, structured output, and media generation
---

# ai

Use this skill when working with the Python `ai` SDK.

```bash
uv add ai
```

Direct OpenAI-compatible and Anthropic-compatible providers require optional
extras: `uv add "ai[openai]"` or `uv add "ai[anthropic]"`. AI Gateway works
with the base package.

```python
import ai
```

## Quick start

```python
import asyncio
import ai


@ai.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72F in {city}"


async def main() -> None:
    model = ai.get_model("gateway:anthropic/claude-sonnet-4")
    agent = ai.agent(tools=[get_weather])

    messages = [
        ai.system_message("You are a helpful weather assistant."),
        ai.user_message("What's the weather in Tokyo?"),
    ]

    async with agent.run(model, messages) as stream:
        async for event in stream:
            if isinstance(event, ai.events.TextDelta):
                print(event.chunk, end="", flush=True)

    print(stream.output)


if __name__ == "__main__":
    asyncio.run(main())
```

`ai.stream(...)` and `agent.run(...)` are async context managers. Iterate events
inside the context. After iteration, read final state from the stream object.

## Models and providers

```python
model = ai.get_model()  # reads AI_SDK_DEFAULT_MODEL
model = ai.get_model("anthropic/claude-sonnet-4")  # unprefixed: gateway route
model = ai.get_model("gateway:anthropic/claude-sonnet-4")
model = ai.get_model("openai:gpt-5.4")  # direct provider route
model = ai.get_model("anthropic:claude-sonnet-4-6")
```

- Gateway credentials use `AI_GATEWAY_API_KEY`.
- Direct providers use provider-specific env vars such as `OPENAI_API_KEY` and
  `ANTHROPIC_API_KEY`.
- Use `ai.get_provider(...)` when you need a custom base URL, API key, headers,
  or client.
- Use `await ai.probe(model)` to check credentials and model availability.

```python
provider = ai.get_provider(
    "openai",
    base_url="http://localhost:1234/v1",
    api_key="your_access_token_here",
)
model = ai.Model("local-model", provider=provider)

models = await ai.get_provider("anthropic").list_models()
```

Request-scoped provider options go through `params`:

```python
params = {
    "providerOptions": {
        "gateway": {"sort": "cost"},
        "anthropic": {"speed": "fast"},
    }
}

async with ai.stream(model, messages, params=params) as stream:
    async for event in stream:
        ...
```

## Messages and events

Messages are Pydantic models with typed parts. Use builders for common roles
and parts:

```python
ai.system_message("Be concise.")
ai.user_message("Describe this image:", ai.file_part(image_bytes, media_type="image/png"))
ai.assistant_message(ai.thinking("scratchpad"), "Final answer")
ai.tool_result_part("tc-1", result={"temp": 72}, tool_name="get_weather")
ai.tool_message(tool_call_id="tc-1", result=72, tool_name="get_weather")
```

Common message properties:

- `message.text`, `message.reasoning`.
- `message.tool_calls`, `message.tool_results`.
- `message.builtin_tool_calls`, `message.builtin_tool_returns`.
- `message.files`, `message.images`, `message.videos`.
- `message.get_output()` or `message.get_output(MyModel)`.

Streams and agents yield event objects from `ai.events`:

```python
async with ai.stream(model, messages, tools=tools) as stream:
    async for event in stream:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)
        elif isinstance(event, ai.events.ToolEnd):
            print(event.tool_call.tool_name, event.tool_call.tool_args)
        elif isinstance(event, ai.events.ToolCallResult):
            for result in event.results:
                print(result.tool_name, result.result)
        elif isinstance(event, ai.events.HookEvent):
            print(event.hook.hook_id, event.hook.status)
        elif isinstance(event, ai.events.PartialToolCallResult):
            print(event.label, event.value)
```

After iteration:

```python
stream.message      # final assistant message for ai.stream
stream.messages     # updated agent history for agent.run
stream.text         # text output for ai.stream
stream.output       # text or parsed Pydantic output
stream.tool_calls   # function tool calls from ai.stream
stream.usage        # latest reported usage
```

Serialize and restore history with Pydantic JSON:

```python
encoded = [message.model_dump(mode="json") for message in stream.messages]
restored = [ai.messages.Message.model_validate(item) for item in encoded]
```

## Direct streaming

Use `ai.stream` when you want one model response and will handle any function
tool calls yourself:

```python
async with ai.stream(model, messages, tools=[get_weather.tool]) as stream:
    async for event in stream:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)

for call in stream.tool_calls:
    print(call.tool_name, call.tool_args)
```

Use structured output with a Pydantic model:

```python
import pydantic


class Forecast(pydantic.BaseModel):
    city: str
    temperature: float


async with ai.stream(model, messages, output_type=Forecast) as stream:
    async for event in stream:
        ...

forecast = stream.output
```

## Tools

A function tool is an async Python function decorated with `@ai.tool`. The
function name becomes the tool name, the docstring becomes the description, and
the signature becomes a Pydantic-validated JSON schema.

```python
@ai.tool
async def scan_sector(sector: str, depth: int = 1) -> str:
    """Scan a sector at the requested depth."""
    return f"{sector}: clear at depth {depth}"
```

Use schema-only tools with `ai.stream` when the SDK should not execute them:

```python
tool = ai.Tool(
    kind="function",
    name="get_weather",
    args=ai.tools.FunctionToolArgs(
        description="Get current weather for a city.",
        params={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
)
```

Provider-executed tools run outside your process:

```python
tools = [ai.providers.anthropic.tools.web_search(max_uses=3)]

async with ai.stream(model, messages, tools=tools) as stream:
    async for event in stream:
        if isinstance(event, ai.events.BuiltinToolResult):
            print(event.result.tool_name, event.result.result)
```

Tool validation failures and exceptions become `ToolCallResult` events with
error result parts. The original exception is on `event.exception` for logging.

```python
if isinstance(event, ai.events.ToolCallResult) and event.exception:
    log_exception(event.exception)
```

## Streaming tools

Async-generator tools yield partial values while they run. An aggregator turns
those values into the final tool result the model sees.

```python
@ai.tool
async def draft_reply(topic: str) -> ai.StreamingTextTool:
    """Draft a reply."""
    yield "Checking "
    yield f"records for {topic}."
```

```python
@ai.tool
async def fetch(url: str) -> ai.StreamingStatusTool[str]:
    """Fetch a URL with status updates."""
    yield "connecting"
    yield "downloading"
    yield body  # last yield is the tool result
```

```python
@ai.tool
async def research(topic: str) -> ai.SubAgentTool:
    """Research a topic with a subagent."""
    subagent = ai.agent(tools=[...])
    async with subagent.run(model, [ai.user_message(topic)]) as stream:
        async for event in stream:
            yield event
```

For custom aggregation, annotate an async-generator return type with
`Annotated[AsyncGenerator[T], ai.agents.Aggregate(...)]`. Built-in
aggregators: `ai.agents.ConcatAggregator`, `ai.agents.LastAggregator`, and
`ai.agents.MessageAggregator`.

## Agents

Use an agent when the SDK should execute Python tools, append tool results, and
continue until the assistant returns a final answer.

```python
agent = ai.agent(tools=[get_weather])

async with agent.run(model, messages) as stream:
    async for event in stream:
        if isinstance(event, ai.events.TextDelta):
            print(event.chunk, end="", flush=True)

history = stream.messages
answer = stream.output
```

Pass structured output and provider params through `agent.run`:

```python
async with agent.run(
    model,
    [ai.user_message("Return a JSON forecast.")],
    output_type=Forecast,
    params={"temperature": 0},
) as stream:
    async for event in stream:
        ...

forecast = stream.output
```

## Custom agent loops

Subclass `ai.Agent` and override `loop` for custom scheduling, routing,
logging, persistence, or approval logic.

```python
from collections.abc import AsyncGenerator


class CustomAgent(ai.Agent):
    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        while context.keep_running():
            async with (
                ai.stream(context=context) as stream,
                ai.ToolRunner() as tool_runner,
            ):
                async for event in ai.util.merge(stream, tool_runner.events()):
                    yield event

                    if isinstance(event, ai.events.ToolEnd):
                        tool_call = context.resolve(event.tool_call)
                        tool_runner.schedule(tool_call)

                context.add(stream.message)
                context.add(tool_runner.get_tool_message())
```

Loop helpers: `context.model`, `context.messages`, `context.tools`,
`context.output_type`, `context.params`, `context.resolve(...)`,
`context.keep_running()`, and `context.add(...)`.

## Multi-agent

Use `ai.SubAgentTool` for agent-as-tool workflows. Use `ai.yield_from(...)`
inside custom loops to fan out streams and forward nested events as
`PartialToolCallResult` values with labels.

```python
async with (
    researcher.run(model, research_messages) as research_stream,
    analyst.run(model, analyst_messages) as analyst_stream,
):
    research_text, analyst_text = await asyncio.gather(
        ai.yield_from(
            research_stream,
            label="researcher",
            aggregator=ai.agents.MessageAggregator,
        ),
        ai.yield_from(
            analyst_stream,
            label="analyst",
            aggregator=ai.agents.MessageAggregator,
        ),
    )
```

Route labels in the consumer:

```python
if isinstance(event, ai.events.PartialToolCallResult):
    if event.label == "researcher":
        route_research(event.value)
```

## Hooks

Hooks are runtime suspension points. Tool approvals are the built-in workflow.

```python
@ai.tool(require_approval=True)
async def delete_file(path: str) -> str:
    """Delete a file."""
    ...
```

The default loop gates each call behind an approval hook with label
`approve_{tool_call_id}` and payload `ai.tools.ToolApproval`.

```python
async with agent.run(model, messages) as stream:
    async for event in stream:
        if isinstance(event, ai.events.HookEvent) and event.hook.status == "pending":
            ai.resolve_hook(
                event.hook.hook_id,
                ai.tools.ToolApproval(granted=True, reason="approved"),
            )
```

Resolve with `granted=False` to deny the call and return an error tool result.

Manual hooks block until resolved in live flows:

```python
approval = await ai.hook(
    "approve_send_email",
    payload=ai.tools.ToolApproval,
    metadata={"tool": "send_email"},
)
```

Resolve or cancel from another task, request handler, or UI callback:

```python
ai.resolve_hook("approve_send_email", {"granted": True, "reason": "approved"})
await ai.cancel_hook("approve_send_email", reason="client disconnected")
```

Hooks emit `HookEvent` objects. Their messages use `role="internal"` and contain
`HookPart` values.

Serverless resume flow:

```python
async with agent.run(model, messages) as stream:
    async for event in stream:
        if isinstance(event, ai.events.HookEvent) and event.hook.status == "pending":
            ai.abort_pending_hook(event.hook)
        yield event

persist(stream.messages)

# Later, restore messages, pre-register the resolution, and rerun.
ai.resolve_hook(hook_id, ai.tools.ToolApproval(granted=True, reason="approved"))
```

## MCP

MCP adapters return `AgentTool` objects usable in `ai.agent(...)`.

```python
tools = await ai.mcp.get_http_tools(
    "https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer token"},
    tool_prefix="docs",
)

tools = await ai.mcp.get_stdio_tools(
    "npx",
    "-y",
    "@anthropic/mcp-server-filesystem",
    "/tmp",
    tool_prefix="fs",
)

agent = ai.agent(tools=tools)
```

## AI SDK UI adapter

Use `ai.agents.ui.ai_sdk` to convert between AI SDK UI messages and Python
runtime messages/events.

```python
class ChatRequest(pydantic.BaseModel):
    messages: list[ai.agents.ui.ai_sdk.UIMessage]


@app.post("/chat")
async def chat(request: ChatRequest):
    messages, approvals = ai.agents.ui.ai_sdk.to_messages(request.messages)
    ai.agents.ui.ai_sdk.apply_approvals(approvals)

    async def stream_response():
        async with chat_agent.run(model, messages) as stream:
            async for chunk in ai.agents.ui.ai_sdk.to_sse(stream):
                yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.agents.ui.ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )
```

Use `ai.agents.ui.ai_sdk.to_ui_messages(messages)` to rebuild UI history from
stored runtime messages.

For serverless approvals, monitor `HookEvent` before passing events to `to_sse`
and call `ai.abort_pending_hook(event.hook)` on pending hooks.

## Media generation

Use `ai.generate` for dedicated image and video models:

```python
image_message = await ai.generate(
    ai.get_model("gateway:google/imagen-4.0-generate-001"),
    [ai.user_message("A watercolor mothership over a quiet city.")],
    ai.ImageParams(n=1, aspect_ratio="16:9"),
)

image = image_message.images[0]
```

For video generation, pass `ai.VideoParams(...)` and read `message.videos`.
