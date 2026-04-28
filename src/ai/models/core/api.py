import dataclasses
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Protocol, Self, runtime_checkable

import pydantic

from ... import types
from ...types import integrity
from . import adapters, params
from . import client as client_
from . import model as model_


@dataclasses.dataclass(frozen=True)
class StreamRequest:
    model: model_.Model
    messages: list[types.Message]
    tools: Sequence[types.ToolLike] | None = None
    output_type: type[pydantic.BaseModel] | None = None


@dataclasses.dataclass(frozen=True)
class GenerateRequest:
    model: model_.Model
    messages: list[types.Message]
    params: params.GenerateParams


@runtime_checkable
class StreamExecutor(Protocol):
    def _do_stream(self, request: StreamRequest) -> AsyncGenerator[types.Event]: ...


@runtime_checkable
class GenerateExecutor(Protocol):
    async def _do_generate(self, request: GenerateRequest) -> types.Message: ...


class Executor:
    """Default executor: dispatches to adapters via the local client."""

    async def _do_stream(self, request: StreamRequest) -> AsyncGenerator[types.Event]:
        c = client_.auto_client(request.model)
        fn = adapters.get_stream_adapter(request.model.adapter)
        async for ev in fn(
            c,
            request.model,
            request.messages,
            tools=request.tools,
            output_type=request.output_type,
        ):
            yield ev

    async def _do_generate(self, request: GenerateRequest) -> types.Message:
        c = client_.auto_client(request.model)
        fn = adapters.get_generate_adapter(request.model.adapter)
        return await fn(c, request.model, request.messages, params=request.params)


_default_executor = Executor()


class Stream:
    """Async-iterable wrapper around an adapter's event stream."""

    def __init__(self, gen: AsyncGenerator[types.Event]) -> None:
        self._gen = gen
        self._message: types.Message = types.Message(role="assistant", parts=[])
        self._parts: dict[str, types.Part] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        await self._gen.aclose()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> types.Event:
        event = await self._gen.__anext__()
        updates = self._aggregate_event(event)
        return event.model_copy(update={"message": self._message, **updates})

    @property
    def message(self) -> types.Message:
        return self._message

    @property
    def usage(self) -> types.Usage | None:
        return self._message.usage

    @property
    def text(self) -> str:
        return self._message.text

    @property
    def tool_calls(self) -> list[types.ToolCallPart]:
        return self._message.tool_calls

    @property
    def output(self) -> Any:
        return self._message.output

    def _aggregate_event(self, event: types.Event) -> dict[str, Any]:
        updates: dict[str, Any] = {}

        # grab usage from any event that carries one
        if event.usage is not None:
            self._message.usage = event.usage

        match event:
            case types.TextStart(block_id=bid):
                tp = types.TextPart(id=bid, text="")
                self._message.parts.append(tp)
                self._parts[bid] = tp
            case types.TextDelta(block_id=bid, chunk=c):
                existing_text = self._parts.get(bid)
                if isinstance(existing_text, types.TextPart):
                    existing_text.text += c
            case types.ReasoningStart(block_id=bid):
                rp = types.ReasoningPart(id=bid, text="")
                self._message.parts.append(rp)
                self._parts[bid] = rp
            case types.ReasoningDelta(block_id=bid, chunk=c):
                existing_reasoning = self._parts.get(bid)
                if isinstance(existing_reasoning, types.ReasoningPart):
                    existing_reasoning.text += c
            case types.ReasoningEnd(block_id=bid, signature=sig):
                existing_reasoning = self._parts.get(bid)
                if (
                    isinstance(existing_reasoning, types.ReasoningPart)
                    and sig is not None
                ):
                    existing_reasoning.signature = sig
            case types.ToolStart(tool_call_id=tcid, tool_name=name):
                tcp = types.ToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args="",
                )
                self._message.parts.append(tcp)
                self._parts[tcid] = tcp
            case types.ToolDelta(tool_call_id=tcid, chunk=c):
                existing_tool = self._parts.get(tcid)
                if isinstance(existing_tool, types.ToolCallPart):
                    existing_tool.tool_args += c
            case types.ToolEnd(tool_call_id=tcid):
                existing_tool = self._parts.get(tcid)
                if isinstance(existing_tool, types.ToolCallPart):
                    updates["tool_call"] = existing_tool
            case types.FileEvent(block_id=bid, media_type=mt, data=d, filename=fname):
                fp = types.FilePart(
                    id=bid or types.generate_id(),
                    data=d,
                    media_type=mt,
                    filename=fname,
                )
                self._message.parts.append(fp)
                self._parts[fp.id] = fp
            case _:
                pass

        return updates


def stream(
    model: model_.Model,
    messages: list[types.Message],
    *,
    tools: Sequence[types.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    executor: StreamExecutor = _default_executor,
) -> Stream:
    """Stream an LLM response."""
    messages = integrity.prepare_messages(messages)
    request = StreamRequest(model, messages, tools, output_type)
    return Stream(executor._do_stream(request))


async def generate(
    model: model_.Model,
    messages: list[types.Message],
    params: params.GenerateParams,
    *,
    executor: GenerateExecutor = _default_executor,
) -> types.Message:
    """Generate a non-streaming response (images, video, etc.)."""
    messages = integrity.prepare_messages(messages)
    request = GenerateRequest(model, messages, params)
    return await executor._do_generate(request)


async def check_connection(model: model_.Model) -> bool:
    """Check whether the model's provider is reachable and the model exists."""
    c = client_.auto_client(model)
    return await model.provider.check(c, model)
