# Writing a custom provider

Only implement a custom provider when adding a new upstream API adapter. For
normal app configuration prefer `ai.get_provider(...)`, `ai.get_model(...)`,
or a protocol override (see `https://ai-python.dev/docs/basics/providers.md`).

Providers emit model events. They do not run Python tools. `ai.stream`
collects events into a `Message`. `ai.Agent` adds tool execution, hooks, and
replay.

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
        yield ai.events.StreamEnd(finish_reason="stop")


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

`provider_class_id` is the serialization discriminator used to restore the
concrete class. Optionally set the `handles: ClassVar[tuple[str, ...]]` class
attribute to register an implementation for matching models.dev provider IDs.
Direct provider model IDs use `ai.get_model("handle:model")`; slash-separated
IDs route through AI Gateway.

Implement `ProviderProtocol.generate` when the upstream API has a native
buffered language-model endpoint. Otherwise, `ai.experimental_generate` falls
back to `stream`. Implement `generate_image`, `generate_video`,
`generate_audio`, `embed`, `transcribe`, or `rerank` only for the dedicated
model operations that the provider supports.

Set response metadata on `StreamEnd`. Normalize the upstream API's native
stop reason into the shared vocabulary: `stop`, `length`, `content_filter`,
`tool_call`, `error`, or `other`. Keep the raw value in `provider_metadata`.
Set `response_id` and `response_model` when the upstream API reports them.

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
