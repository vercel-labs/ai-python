import contextlib
import dataclasses
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Generic, Protocol, Self, cast, overload, runtime_checkable

import pydantic

# ``typing.TypeVar`` lacks the ``default=`` kwarg on Python <3.13.
# Use the typing_extensions backport so this works on 3.12 too.
from typing_extensions import TypeVar  # noqa: UP035

from ... import types
from ...types import integrity
from . import adapters
from . import model as model_
from . import params as params_

# Stream output type.  Defaults to ``str``: when the stream was opened
# without an ``output_type``, ``Stream.output`` returns the concatenated
# message text.
StreamOutputT = TypeVar("StreamOutputT", default=str)


@dataclasses.dataclass(frozen=True)
class StreamRequest:
    model: model_.Model
    messages: list[types.messages.Message]
    tools: Sequence[types.tools.Tool] | None = None
    output_type: type[pydantic.BaseModel] | None = None
    params: Any = None


@dataclasses.dataclass(frozen=True)
class GenerateRequest:
    model: model_.Model
    messages: list[types.messages.Message]
    params: params_.GenerateParams


@runtime_checkable
class StreamExecutor(Protocol):
    def _do_stream(
        self,
        request: StreamRequest,
    ) -> AsyncGenerator[types.events.Event]: ...


@runtime_checkable
class GenerateExecutor(Protocol):
    async def _do_generate(
        self, request: GenerateRequest
    ) -> types.messages.Message: ...


class Executor:
    """Default executor: dispatches to adapters via the local client."""

    async def _do_stream(
        self,
        request: StreamRequest,
    ) -> AsyncGenerator[types.events.Event]:
        fn = adapters.get_stream_adapter(request.model.adapter)
        kwargs: dict[str, Any] = {}
        if request.params is not None:
            kwargs["params"] = request.params
        async for ev in fn(
            request.model,
            request.messages,
            tools=request.tools,
            output_type=request.output_type,
            **kwargs,
        ):
            yield ev

    async def _do_generate(self, request: GenerateRequest) -> types.messages.Message:
        fn = adapters.get_generate_adapter(request.model.adapter)
        return await fn(request.model, request.messages, params=request.params)


_default_executor = Executor()


class Stream(Generic[StreamOutputT]):
    """Async-iterable wrapper around an adapter's event stream."""

    def __init__(
        self,
        gen: AsyncGenerator[types.events.Event],
        *,
        seed_message: types.messages.Message | None = None,
        output_type: type[StreamOutputT] | None = None,
    ) -> None:
        """Wrap an event generator.

        ``seed_message`` seeds the in-progress assistant message. Pass
        a copy of an existing turn when replaying so
        ``stream.message`` ends up identical to that turn instead of
        being rebuilt from synthetic events.  When ``None`` (default),
        an empty assistant message is created and rebuilt from the
        incoming events.

        ``output_type`` is the Pydantic model the request was constrained
        to.  When set, ``Stream.output`` validates the streamed JSON text
        against it.  When ``None`` (default), ``Stream.output`` returns
        the concatenated text content unchanged.
        """
        self._gen = gen
        self._message: types.messages.Message = seed_message or types.messages.Message(
            role="assistant", parts=[]
        )
        self._parts: dict[str, types.messages.Part] = {}
        # ``output_type`` is typed against the public ``StreamOutputT`` type
        # param for ergonomics; internally we know it's a Pydantic model
        # subclass (or None for the text-default case).
        self._output_type = cast(type[pydantic.BaseModel] | None, output_type)

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
    def output(self) -> StreamOutputT:
        """Return the streamed output as the ``output_type`` passed in.

        Defaults to the concatenated message text.  When a Pydantic
        model subclass was passed, validates the streamed JSON against
        it and returns the parsed instance.
        """
        return cast(StreamOutputT, self._message.get_output(self._output_type))

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


@runtime_checkable
class StreamContext(Protocol):
    """Anything that exposes the fields :func:`stream` reads off a context.

    Used to let callers pass an ``agents.Context`` to :func:`stream`
    without an import-time circular dependency.
    """

    @property
    def model(self) -> model_.Model: ...
    @property
    def messages(self) -> list[types.messages.Message]: ...
    @property
    def tools(self) -> list[types.tools.Tool]: ...
    @property
    def output_type(self) -> type[pydantic.BaseModel] | None: ...
    @property
    def params(self) -> Any: ...


@overload
def stream(
    *,
    context: StreamContext,
    params: Any = None,
    executor: StreamExecutor = _default_executor,
) -> AbstractAsyncContextManager[Stream[str]]: ...
@overload
def stream[T: pydantic.BaseModel](
    *,
    context: StreamContext,
    output_type: type[T],
    params: Any = None,
    executor: StreamExecutor = _default_executor,
) -> AbstractAsyncContextManager[Stream[T]]: ...
@overload
def stream(
    model: model_.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    params: Any = None,
    executor: StreamExecutor = _default_executor,
) -> AbstractAsyncContextManager[Stream[str]]: ...
@overload
def stream[T: pydantic.BaseModel](
    model: model_.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[T],
    params: Any = None,
    executor: StreamExecutor = _default_executor,
) -> AbstractAsyncContextManager[Stream[T]]: ...
def stream(
    model: model_.Model | None = None,
    messages: list[types.messages.Message] | None = None,
    *,
    context: StreamContext | None = None,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: Any = None,
    executor: StreamExecutor = _default_executor,
) -> AbstractAsyncContextManager[Stream[Any]]:
    """Stream an LLM response.

    Used as an async context manager whose value is the :class:`Stream`.
    Pass either positional ``model, messages`` (plus optional ``tools=``)
    or ``context=`` (an ``agents.Context`` or anything matching
    :class:`StreamContext`)::

        async with ai.stream(model, messages) as s: ...
        async with ai.stream(context=context) as s: ...

    If the last message is marked ``replay=True``, replay that turn as
    synthetic stream events instead of calling the model.
    """
    if context is not None:
        if model is not None or messages is not None or tools is not None:
            raise TypeError(
                "stream() takes either model/messages/tools or context=, not both"
            )
        model = context.model
        messages = context.messages
        tools = context.tools
        if output_type is None:
            output_type = context.output_type
        if params is None:
            params = context.params
    elif model is None or messages is None:
        raise TypeError("stream() requires either model and messages or context=")

    return _stream(
        model=model,
        messages=messages,
        tools=tools,
        output_type=output_type,
        params=params,
        executor=executor,
    )


@contextlib.asynccontextmanager
async def _stream(
    *,
    model: model_.Model,
    messages: list[types.messages.Message],
    tools: Sequence[types.tools.Tool] | None,
    output_type: type[pydantic.BaseModel] | None,
    params: Any,
    executor: StreamExecutor,
) -> AsyncIterator[Stream[Any]]:
    if messages and messages[-1].replay:
        last = messages[-1]
        s = Stream(
            _replay_tool_calls(last),
            seed_message=last.model_copy(deep=True),
            output_type=output_type,
        )
    else:
        prepared = integrity.prepare_messages(messages)
        request = StreamRequest(
            model,
            prepared,
            tools,
            output_type,
            params,
        )
        s = Stream(executor._do_stream(request), output_type=output_type)
    try:
        yield s
    finally:
        await s.aclose()


async def generate(
    model: model_.Model,
    messages: list[types.messages.Message],
    params: params_.GenerateParams,
    *,
    executor: GenerateExecutor = _default_executor,
) -> types.messages.Message:
    """Generate a non-streaming response (images, video, etc.)."""
    messages = integrity.prepare_messages(messages)
    request = GenerateRequest(model, messages, params)
    return await executor._do_generate(request)


async def probe(model: model_.Model) -> None:
    """Raise unless the model's provider is reachable and the model exists."""
    await model.provider.probe(model)
