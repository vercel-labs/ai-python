---
name: ai-python-durable-execution
description: Use when adding durable execution to AI SDK for Python, building durable agent loops, or serializing messages across workflow steps.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-durable-execution

Use durable execution when an agent run must survive restarts, worker moves, or
long waits.

The SDK does not provide durability by itself. Build a custom `Agent.loop`, and
put side effects inside durable steps:

- model calls
- tool I/O
- approval or resume boundaries

Keep the workflow replayable. Durable steps should take JSON inputs and return
JSON outputs.

Serialize messages like this:

```python
data = message.model_dump(mode="json")
message = ai.messages.Message.model_validate(data)
```

## Model Step

A durable model step should drain `ai.stream(...)` inside the step and return one
complete assistant `Message`.

```python
@workflow.step
async def llm_step(
    model_data: dict[str, object],
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
) -> dict[str, object]:
    model = ai.Model.model_validate(model_data)
    messages = [
        ai.messages.Message.model_validate(message)
        for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]

    async with ai.stream(model, messages, tools=tools) as stream:
        async for _event in stream:
            pass

        if stream.message is None:
            raise RuntimeError("LLM stream ended without a message")

        return stream.message.model_dump(mode="json")
```

## Durable Tools

Prefer wrapping the tool body in the durable step:

```python
@ai.tool
@workflow.step
async def ask_mothership(question: str) -> str:
    response = await mothership_client.ask(question)
    return response.summary
```

If the workflow system needs separate activity dispatch, schedule a zero-arg
callable that returns `ai.tool_result(...)`. Do not call `tool.fn` directly.

## Agent Loop

Use the model step result as a complete message. Do not wrap it in `ai.Stream`,
`ai.events.replay_message_events`, or `ai.util.merge`, those utilities are
used for fluent dispatch in non-durable applications, which is impossible
in a workflow setting since streams are considered side-effects.

```python
class DurableAgent(ai.Agent):
    async def loop(self, context: ai.Context):
        while context.keep_running():
            result = await llm_step(
                context.model.model_dump(mode="json"),
                [m.model_dump(mode="json") for m in context.messages],
                [t.model_dump(mode="json") for t in context.tools],
            )

            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)

            async with ai.ToolRunner() as runner:
                for tool_call in assistant_message.tool_calls:
                    runner.schedule(context.resolve(tool_call))

                async for event in runner.events():
                    yield event

                context.add(runner.get_tool_message())
```

This pattern does not stream model tokens to the caller. That is usually the
right tradeoff for durable workflows, because many durable systems do not support
async generators. You can build a queue-based side channel for streaming; however,
that kind of stream can't be used to dispatch tools and affect control flow directly.
