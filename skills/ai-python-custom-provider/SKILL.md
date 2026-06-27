---
name: ai-python-custom-provider
description: Use for implementing custom providers in AI SDK for Python.
metadata:
  sdk-version: "0.2.1"
---

# ai-python-custom-provider

Providers emit model events. They do not run Python tools. `ai.stream` collects
events into a `Message`. `ai.Agent` adds tool execution, hooks, and replay.

Minimal shape:

```python
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal

import pydantic
import ai


class MyProtocol(ai.ProviderProtocol[Any]):
    protocol_class_id: Literal["my_protocol"] = "my_protocol"

    def stream(
        self,
        client: Any,
        model: ai.Model,
        messages: list[ai.messages.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: ai.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[ai.events.Event]:
        return self._stream(client, model, messages, tools=tools)

    async def _stream(
        self,
        client: Any,
        model: ai.Model,
        messages: list[ai.messages.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None,
    ) -> AsyncGenerator[ai.events.Event]:
        yield ai.events.StreamStart()
        yield ai.events.TextStart(block_id="text")
        yield ai.events.TextDelta(block_id="text", chunk="Hello")
        yield ai.events.TextEnd(block_id="text")
        yield ai.events.StreamEnd()


class MyProvider(ai.Provider[Any]):
    provider_class_id: Literal["my_provider"] = "my_provider"
    name: str = "my"
    default_base_url: str = "https://example.invalid"

    def __init__(self, *, client: Any) -> None:
        super().__init__()
        self._set_client(client)

    def default_protocol(self) -> ai.ProviderProtocol[Any]:
        return MyProtocol()

    async def list_models(self) -> list[str]:
        return ["my-model"]

    async def probe(self, model: ai.Model) -> None:
        return None


model = ai.Model(id="my-model", provider=MyProvider(client=client))
```

For Python tool calls, emit `ToolStart`, `ToolDelta`, and `ToolEnd`:

```python
yield ai.events.ToolStart(tool_call_id=tcid, tool_name=name)
yield ai.events.ToolDelta(tool_call_id=tcid, chunk=args_json)
yield ai.events.ToolEnd(
    tool_call_id=tcid,
    tool_call=ai.messages.DUMMY_TOOL_CALL,
)
```

The stream collector fills `event.tool_call` with the aggregated tool call.
Then `Agent` resolves and runs the tool.

If the provider runs its own built-in tool, emit `BuiltinToolStart`,
`BuiltinToolDelta`, `BuiltinToolEnd`, and `BuiltinToolResult` instead.

Do not implement a custom provider for normal app configuration. Prefer
`ai.get_provider(...)`, `ai.get_model(...)`, or a protocol override unless you
are adding a new upstream API adapter.
