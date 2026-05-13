---
name: ai
description: Python `ai` module — models, agents, hooks, MCP, structured output
---

# ai

```bash
uv add vercel-ai-sdk
```

```python
import ai
```

## Quick reference

```python
@ai.tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72F in {city}"

model = ai.ai_gateway("anthropic/claude-sonnet-4")
agent = ai.agent(tools=[get_weather])

messages = [
    ai.system_message("You are a helpful weather assistant."),
    ai.user_message("What's the weather in Tokyo?"),
]

async for msg in agent.run(model, messages):
    print(msg.text_delta, end="")
```

`ai.openai(model_id)`, `ai.anthropic(model_id)`, `ai.ai_gateway(model_id)` — provider factories, callable, return `Model`. Clients auto-created from env vars (`AI_GATEWAY_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Pass `ai.Client(base_url=, api_key=)` to the provider call for custom endpoints: `ai.openai("gpt-5.4", client=c)`. `provider.list()` returns available model IDs.

`ai.stream(model, messages, ...)` — streaming without an agent loop. Returns `StreamResult` with `.text`, `.tool_calls`, `.output`, `.usage` after iteration.

`ai.generate(model, messages, params)` — non-streaming generation. `params` is `ImageParams` or `VideoParams`.

`ai.check_connection(model)` — verify credentials and model availability.

## Messages

Immutable Pydantic models. Use builders:

```python
ai.system_message("Be concise.")
ai.user_message("Describe this image:", ai.file_part(url))
ai.assistant_message(...)
ai.tool_message(...)       # merge one or more tool-result messages/parts
ai.tool_result(...)        # single ToolResultPart
ai.thinking(...)
```

Key properties on streamed messages:

- `msg.text_delta` — current text chunk (for live display)
- `msg.text` — full accumulated text
- `msg.tool_calls` — list of `ToolCallPart` on assistant messages
- `msg.output` — validated Pydantic instance (when using `output_type`)
- `msg.get_hook_part()` — find a hook suspension part

Serialize: `msg.model_dump()`. Restore: `ai.messages.Message.model_validate(data)`.

## Streaming tools

A regular `@ai.tool` returns the tool result directly. An async-generator tool yields intermediate values that stream to the consumer as `ai.events.PartialToolCallResult` events; an *aggregator* decides what the model sees as the final result.

Declare the aggregator via the return-type annotation:

```python
@ai.tool
async def fetch(url: str) -> ai.StreamingStatusTool[str]:
    yield "connecting..."
    yield "downloading..."
    yield body              # last yield = tool result; earlier yields = status

@ai.tool
async def render(prompt: str) -> ai.StreamingTextTool:
    async for chunk in stream:
        yield chunk         # all chunks concatenated into the tool result

@ai.tool
async def research(topic: str) -> ai.SubAgentTool:
    sub = ai.agent(tools=[...])
    async for event in sub.run(model, msgs):
        yield event         # final assistant text becomes the tool result
```

These aliases expand to `Annotated[AsyncGenerator[T], ai.agents.Aggregate(Aggregator)]`. For a parameterized aggregator, write the marker directly:

```python
type Joined = Annotated[AsyncGenerator[str], ai.agents.Aggregate(ai.agents.ConcatAggregator, delim="\n")]
```

Or pass the factory as a keyword argument: `@ai.tool(aggregator=ai.agents.LastAggregator)`. Specifying both an annotation marker and the kwarg raises `TypeError`.

Built-in aggregators: `ai.agents.ConcatAggregator`, `ai.agents.LastAggregator`, `ai.agents.MessageAggregator`. Subclass `ai.agents.SimpleAggregator[Item, Result]` (in/out same type) or `ai.events.Aggregator[Item, Result, ModelResult]` (separate model output) for custom ones.

## Custom agent loops

Override the default loop when you need approval gates, routing, or batching.

The loop yields `AgentEvent`s for the consumer and updates history with `context.add(message)`. Anything not added to `context.messages` won't be visible on the next turn.

**Concurrent form** — fan tools out as the model emits them, mirrors the default loop.

```python
my_agent = ai.agent(tools=[get_weather, get_population])

@my_agent.loop
async def custom(context: ai.Context):
    while context.keep_running():
        async with (
            ai.stream(
                context.model, context.messages, tools=context.tools
            ) as s,
            ai.ToolRunner(s) as tr,
        ):
            async for event in ai.util.merge(s, tr.events()):
                yield event
                if isinstance(event, ai.events.ToolEnd):
                    tr.schedule(context.resolve(event.tool_call))

            context.add(s.message)
            context.add(tr.get_tool_message())
```

**Sequential form** — drain the stream, then run tools. Simpler when you need to gate tool calls before executing them:

```python
@my_agent.loop
async def custom(context: ai.Context):
    while True:
        s = ai.stream(
            context.model, context.messages, tools=context.tools
        )
        async for event in s:
            yield event
        context.add(s.message)

        tool_calls = context.resolve(s.tool_calls)
        if not tool_calls:
            return

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(tc()) for tc in tool_calls]

        context.add(ai.tool_message(*(t.result() for t in tasks)))
```

Loop helpers: `context.model`, `context.messages`, `context.tools`, `context.resolve(s.tool_calls)`, `context.keep_running()`, `context.add(msg)`. `await tc()` executes a tool call and returns a `ToolCallResult`; pass those (or `Message`s) to `ai.tool_message(...)` to merge into one tool-result message.

## Multi-agent

Use `asyncio.gather` with `ai.yield_from(...)` and labels to run agents in parallel:

```python
async def multi(model: ai.Model, query: str) -> str:
    researcher = ai.agent(tools=[t1])
    analyst = ai.agent(tools=[t2])

    r1, r2 = await asyncio.gather(
        ai.yield_from(
            researcher.run(model, msgs1),
            label="researcher",
            aggregator=ai.agents.MessageAggregator,
        ),
        ai.yield_from(
            analyst.run(model, msgs2),
            label="analyst",
            aggregator=ai.agents.MessageAggregator,
        ),
    )
    return f"{r1}\n{r2}"
```

Each forwarded event is wrapped in `ai.events.PartialToolCallResult` carrying the `label`, so the consumer can route by source.

## Hooks

Typed suspension points for human-in-the-loop.

**Tool approval (built-in shortcut).** Flag a tool with `require_approval=True` and the default loop gates each call behind an `ai.tools.ToolApproval` hook (label `approve_{tool_call_id}`, payload carries `granted` + `reason`):

```python
@ai.tool(require_approval=True)
async def delete_file(path: str) -> str:
    ...

# consumer-side resolve:
ai.resolve_hook(hook.hook_id, ai.tools.ToolApproval(granted=True))
```

Denial returns an error `ToolCallResult` with `result=f"Rejected: {reason}"`. For custom payloads, label schemes, or per-call gating, write a custom loop using the primitives below.

**Manual hooks.** Inside agent code (blocks until resolved):

```python
class Approval(pydantic.BaseModel):
    granted: bool
    reason: str

approval = await ai.hook(
    "approve_send_email",
    payload=Approval,
    metadata={"tool": "send_email"},
)
if approval.granted:
    await tc()
```

From outside:

```python
ai.resolve_hook("approve_send_email", {"granted": True, "reason": "User approved"})
await ai.cancel_hook("approve_send_email")
```

Hook messages have `role="signal"` with a `HookPart`.

**Long-running mode** (`interrupt_loop=False`, default): await blocks until resolved. Use for websocket/interactive UIs.

**Serverless mode** (`interrupt_loop=True`): unresolved hooks cancel the run. Pre-register a resolution with `ai.resolve_hook(...)` before rerunning.

Consuming hooks in the iterator:

```python
async for msg in my_agent.run(model, messages):
    if msg.role == "signal" and (hook := msg.get_hook_part()):
        answer = input(f"Approve {hook.hook_id}? [y/n] ")
        ai.resolve_hook(
            hook.hook_id,
            Approval(granted=answer == "y", reason="operator"),
        )
        continue
    print(msg.text_delta, end="")
```

## Structured output

```python
class Forecast(pydantic.BaseModel):
    city: str
    temperature: float

stream = await ai.stream(model, messages, output_type=Forecast)
async for msg in stream:
    ...
stream.output.city
```

## MCP

```python
tools = await ai.mcp.get_http_tools(
    "https://mcp.example.com/mcp",
    headers={...},
    tool_prefix="docs",
)
tools = await ai.mcp.get_stdio_tools(
    "npx", "-y", "@anthropic/mcp-server-filesystem", "/tmp",
    tool_prefix="fs",
)
```

Returns `Tool` objects usable in `ai.stream(...)` or `ai.agent(...)`.

## AI SDK UI adapter

```python
from ai.ai_sdk_ui import UI_MESSAGE_STREAM_HEADERS, to_messages, to_sse_stream

messages = to_messages(request.messages)
return StreamingResponse(
    to_sse_stream(agent.run(model, messages)),
    headers=UI_MESSAGE_STREAM_HEADERS,
)
```

