import contextlib
import dataclasses
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any, Protocol, Self, cast, runtime_checkable

import pydantic

from ... import types
from ...types import integrity
from . import adapters
from . import client as client_
from . import model as model_
from . import params as params_


@dataclasses.dataclass(frozen=True)
class StreamRequest[ProviderParamsT: pydantic.BaseModel]:
    model: model_.Model[ProviderParamsT]
    messages: list[types.messages.Message]
    tools: Sequence[types.tools.Tool] | None = None
    output_type: type[pydantic.BaseModel] | None = None
    params: params_.StreamParams[ProviderParamsT] | None = None


@dataclasses.dataclass(frozen=True)
class GenerateRequest:
    model: model_.Model[Any]
    messages: list[types.messages.Message]
    params: params_.GenerateParams


@runtime_checkable
class StreamExecutor(Protocol):
    def _do_stream[ProviderParamsT: pydantic.BaseModel](
        self,
        request: StreamRequest[ProviderParamsT],
    ) -> AsyncGenerator[types.events.Event]: ...


@runtime_checkable
class GenerateExecutor(Protocol):
    async def _do_generate(
        self, request: GenerateRequest
    ) -> types.messages.Message: ...


class Executor:
    """Default executor: dispatches to adapters via the local client."""

    async def _do_stream[ProviderParamsT: pydantic.BaseModel](
        self,
        request: StreamRequest[ProviderParamsT],
    ) -> AsyncGenerator[types.events.Event]:
        c = client_.auto_client(request.model)
        fn = adapters.get_stream_adapter(request.model.adapter)
        kwargs: dict[str, Any] = {}
        if request.params is not None:
            kwargs["params"] = request.params
        async for ev in fn(
            c,
            request.model,
            request.messages,
            tools=request.tools,
            output_type=request.output_type,
            **kwargs,
        ):
            yield ev

    async def _do_generate(self, request: GenerateRequest) -> types.messages.Message:
        c = client_.auto_client(request.model)
        fn = adapters.get_generate_adapter(request.model.adapter)
        return await fn(c, request.model, request.messages, params=request.params)


_default_executor = Executor()


# this is here because of the string-based registry for model apis
# that erases type information
def _normalize_stream_params[ProviderParamsT: pydantic.BaseModel](
    model: model_.Model[ProviderParamsT],
    value: params_.StreamParams[ProviderParamsT] | None,
) -> params_.StreamParams[ProviderParamsT] | None:
    if value is None:
        return None
    if isinstance(value, pydantic.BaseModel):
        raw_items: list[Any] = [value]
    else:
        raw_items = list(value)

    expected_type = getattr(model.provider, "params_type", pydantic.BaseModel)
    normalized: list[ProviderParamsT] = []
    for item in raw_items:
        if not isinstance(item, pydantic.BaseModel):
            raise TypeError(
                "stream params must be pydantic BaseModel instances with a "
                "provider params type"
            )
        if not isinstance(item, expected_type):
            if isinstance(expected_type, tuple):
                expected = " or ".join(type_.__name__ for type_ in expected_type)
            else:
                expected = expected_type.__name__
            raise TypeError(
                f"{model.provider.name} streams accept "
                f"{expected}, got {type(item).__name__}"
            )
        normalized.append(cast(ProviderParamsT, item))

    if not normalized:
        return None
    if len(normalized) == 1:
        return normalized[0]
    return normalized


class Stream:
    """Async-iterable wrapper around an adapter's event stream."""

    def __init__(
        self,
        gen: AsyncGenerator[types.events.Event],
        *,
        seed_message: types.messages.Message | None = None,
    ) -> None:
        """Wrap an event generator.

        ``seed_message`` seeds the in-progress assistant message. Pass
        a copy of an existing turn when replaying so
        ``stream.message`` ends up identical to that turn instead of
        being rebuilt from synthetic events.  When ``None`` (default),
        an empty assistant message is created and rebuilt from the
        incoming events.
        """
        self._gen = gen
        self._message: types.messages.Message = seed_message or types.messages.Message(
            role="assistant", parts=[]
        )
        self._parts: dict[str, types.messages.Part] = {}

    async def aclose(self) -> None:
        await self._gen.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool:
        await self.aclose()
        return False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self: Self) -> types.events.Event:
        try:
            event = await self._gen.__anext__()
        except Exception:
            # Usually this fires on StopAsyncIteration, but could be a
            # real exception too
            raise
        updates = self._aggregate_event(event)
        return event.model_copy(update={"message": self._message, **updates})

    @property
    def message(self) -> types.messages.Message:
        return self._message

    @property
    def usage(self) -> types.usage.Usage | None:
        return self._message.usage

    @property
    def text(self) -> str:
        return self._message.text

    @property
    def tool_calls(self) -> list[types.messages.ToolCallPart]:
        return self._message.tool_calls

    @property
    def output(self) -> Any:
        return self._message.output

    def _aggregate_event(self, event: types.events.Event) -> dict[str, Any]:
        updates: dict[str, Any] = {}

        # Replay events carry no new state — the seeded message already
        # has everything they would have produced.
        if event.replay:
            return updates

        # grab usage from any event that carries one
        if event.usage is not None:
            self._message.usage = event.usage

        match event:
            case types.events.TextStart(block_id=bid, provider_metadata=pm):
                tp = types.messages.TextPart(id=bid, text="", provider_metadata=pm)
                self._message.parts.append(tp)
                self._parts[bid] = tp
            case types.events.TextDelta(block_id=bid, chunk=c, provider_metadata=pm):
                existing_text = self._parts.get(bid)
                if isinstance(existing_text, types.messages.TextPart):
                    existing_text.text += c
                    if pm is not None:
                        existing_text.provider_metadata = pm
            case types.events.TextEnd(block_id=bid, provider_metadata=pm):
                existing_text = self._parts.get(bid)
                if (
                    isinstance(existing_text, types.messages.TextPart)
                    and pm is not None
                ):
                    existing_text.provider_metadata = pm
            case types.events.ReasoningStart(block_id=bid, provider_metadata=pm):
                rp = types.messages.ReasoningPart(id=bid, text="", provider_metadata=pm)
                self._message.parts.append(rp)
                self._parts[bid] = rp
            case types.events.ReasoningDelta(
                block_id=bid, chunk=c, provider_metadata=pm
            ):
                existing_reasoning = self._parts.get(bid)
                if isinstance(existing_reasoning, types.messages.ReasoningPart):
                    existing_reasoning.text += c
                    if pm is not None:
                        existing_reasoning.provider_metadata = pm
            case types.events.ReasoningEnd(block_id=bid, provider_metadata=pm):
                existing_reasoning = self._parts.get(bid)
                if (
                    isinstance(existing_reasoning, types.messages.ReasoningPart)
                    and pm is not None
                ):
                    existing_reasoning.provider_metadata = pm
            case types.events.ToolStart(
                tool_call_id=tcid, tool_name=name, provider_metadata=pm
            ):
                tcp = types.messages.ToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args="",
                    provider_metadata=pm,
                )
                self._message.parts.append(tcp)
                self._parts[tcid] = tcp
            case types.events.ToolDelta(
                tool_call_id=tcid, chunk=c, provider_metadata=pm
            ):
                existing_tool = self._parts.get(tcid)
                if isinstance(existing_tool, types.messages.ToolCallPart):
                    existing_tool.tool_args += c
                    if pm is not None:
                        existing_tool.provider_metadata = pm

            case types.events.ToolEnd(tool_call_id=tcid, provider_metadata=pm):
                existing_tool = self._parts.get(tcid)
                if isinstance(existing_tool, types.messages.ToolCallPart):
                    updates["tool_call"] = existing_tool
                    if pm is not None:
                        existing_tool.provider_metadata = pm
            case types.events.BuiltinToolStart(
                tool_call_id=tcid,
                tool_name=name,
                provider_metadata=pm,
            ):
                btcp = types.messages.BuiltinToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args="",
                    provider_metadata=pm,
                )
                self._message.parts.append(btcp)
                self._parts[tcid] = btcp
            case types.events.BuiltinToolDelta(
                tool_call_id=tcid, chunk=c, provider_metadata=pm
            ):
                existing_btc = self._parts.get(tcid)
                if isinstance(existing_btc, types.messages.BuiltinToolCallPart):
                    existing_btc.tool_args += c
                    if pm is not None:
                        existing_btc.provider_metadata = pm
            case types.events.BuiltinToolEnd(tool_call_id=tcid, provider_metadata=pm):
                existing_btc = self._parts.get(tcid)
                if isinstance(existing_btc, types.messages.BuiltinToolCallPart):
                    updates["tool_call"] = existing_btc
                    if pm is not None:
                        existing_btc.provider_metadata = pm
            case types.events.BuiltinToolResult(result=res, provider_metadata=pm):
                if pm is not None:
                    res = res.model_copy(update={"provider_metadata": pm})
                self._message.parts.append(res)
            case types.events.FileEvent(
                block_id=bid,
                media_type=mt,
                data=d,
                filename=fname,
                provider_metadata=pm,
            ):
                fp = types.messages.FilePart(
                    id=bid or types.messages.generate_id(),
                    data=d,
                    media_type=mt,
                    filename=fname,
                    provider_metadata=pm,
                )
                self._message.parts.append(fp)
                self._parts[fp.id] = fp

            case types.events.StreamEnd(provider_metadata=pm):
                if pm is not None:
                    self._message.provider_metadata = pm
            case _:
                pass

        return updates


async def _replay_tool_calls(
    msg: types.messages.Message,
) -> AsyncGenerator[types.events.Event]:
    """Replay an assistant turn's tool calls as ``replay``-flagged ``ToolEnd``.

    Used by :func:`stream` to short-circuit when the last message is
    already marked for replay — letting resume flows (e.g. post-hook
    re-entry) re-dispatch the existing tool calls without hitting the
    LLM and without re-streaming the original text/reasoning to the
    consumer.  The wrapping :class:`Stream`'s ``message`` is seeded
    with the original turn so callers see the same parts they would
    have without replay.
    """
    for part in msg.tool_calls:
        yield types.events.ToolEnd(
            tool_call_id=part.tool_call_id,
            tool_call=part,
            replay=True,
        )


@contextlib.asynccontextmanager
async def stream[ProviderParamsT: pydantic.BaseModel](
    model: model_.Model[ProviderParamsT],
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: params_.StreamParams[ProviderParamsT] | None = None,
    executor: StreamExecutor = _default_executor,
) -> AsyncIterator[Stream]:
    """Stream an LLM response.

    Used as an async context manager whose value is the :class:`Stream`::

        async with ai.stream(model, messages) as s:
            async for event in s:
                ...
            print(s.message)

    If the last message is marked ``replay=True``, replay that turn as
    synthetic stream events instead of calling the model.
    """
    if messages and messages[-1].replay:
        last = messages[-1]
        s = Stream(_replay_tool_calls(last), seed_message=last.model_copy(deep=True))
    else:
        prepared = integrity.prepare_messages(messages)
        stream_params = _normalize_stream_params(model, params)
        request = StreamRequest[ProviderParamsT](
            model,
            prepared,
            tools,
            output_type,
            stream_params,
        )
        s = Stream(executor._do_stream(request))
    try:
        yield s
    finally:
        await s.aclose()


async def generate(
    model: model_.Model[Any],
    messages: list[types.messages.Message],
    params: params_.GenerateParams,
    *,
    executor: GenerateExecutor = _default_executor,
) -> types.messages.Message:
    """Generate a non-streaming response (images, video, etc.)."""
    messages = integrity.prepare_messages(messages)
    request = GenerateRequest(model, messages, params)
    return await executor._do_generate(request)


async def check_connection(model: model_.Model[Any]) -> bool:
    """Check whether the model's provider is reachable and the model exists."""
    c = client_.auto_client(model)
    return await model.provider.check(c, model)
