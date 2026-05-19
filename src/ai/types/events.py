import abc
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Annotated, Any, Literal

import pydantic

from . import messages
from . import usage as usage_

# we're using pydantic because events are crossing
# serialization border in the case of durable execution


# Placeholder so BaseEvent.message is typed as Message (not Message | None).
# Stream.__anext__ stamps the real in-progress message before yielding,
# so consumers never see this value.
_DUMMY_MESSAGE = messages.Message(id="<unset>", role="assistant", parts=[])


class BaseEvent(pydantic.BaseModel):
    """Common fields stamped onto every event by the streaming wrapper.

    ``message`` carries the in-progress (or final) assistant message; the
    streaming layer aggregates parts into it as deltas arrive and stamps
    a reference onto each yielded event. ``usage`` carries the latest
    usage value reported by the provider (latest-wins across the stream).

    ``replay`` is set on synthetic events emitted when ``models.stream``
    short-circuits an existing assistant turn (resume-after-approval
    flows).  ``Agent.run`` drops replay-flagged events from the consumer-
    facing stream — the loop's tool dispatcher still consumes them
    internally.  Excluded from JSON: it's a control flag, not data.
    """

    message: messages.Message = _DUMMY_MESSAGE
    usage: usage_.Usage | None = None
    provider_metadata: dict[str, Any] | None = None
    replay: bool = pydantic.Field(default=False, exclude=True, repr=False)

    model_config = pydantic.ConfigDict(frozen=True)


class StreamStart(BaseEvent):
    kind: Literal["stream_start"] = "stream_start"


class StreamEnd(BaseEvent):
    kind: Literal["stream_end"] = "stream_end"


class TextStart(BaseEvent):
    block_id: str = ""

    kind: Literal["text_start"] = "text_start"


class TextDelta(BaseEvent):
    chunk: str
    block_id: str = ""

    kind: Literal["text_delta"] = "text_delta"


class TextEnd(BaseEvent):
    block_id: str = ""

    kind: Literal["text_end"] = "text_end"


class ReasoningStart(BaseEvent):
    block_id: str = ""

    kind: Literal["reasoning_start"] = "reasoning_start"


class ReasoningDelta(BaseEvent):
    chunk: str
    block_id: str = ""

    kind: Literal["reasoning_delta"] = "reasoning_delta"


class ReasoningEnd(BaseEvent):
    block_id: str = ""

    kind: Literal["reasoning_end"] = "reasoning_end"


class ToolStart(BaseEvent):
    tool_call_id: str = ""
    tool_name: str = ""

    kind: Literal["tool_start"] = "tool_start"


class ToolDelta(BaseEvent):
    chunk: str
    tool_call_id: str = ""

    kind: Literal["tool_delta"] = "tool_delta"


class ToolEnd(BaseEvent):
    tool_call: messages.ToolCallPart
    tool_call_id: str = ""

    kind: Literal["tool_end"] = "tool_end"


class BuiltinToolStart(BaseEvent):
    tool_call_id: str = ""
    tool_name: str = ""

    kind: Literal["builtin_tool_start"] = "builtin_tool_start"


class BuiltinToolDelta(BaseEvent):
    chunk: str
    tool_call_id: str = ""

    kind: Literal["builtin_tool_delta"] = "builtin_tool_delta"


class BuiltinToolEnd(BaseEvent):
    tool_call: messages.BuiltinToolCallPart
    tool_call_id: str = ""

    kind: Literal["builtin_tool_end"] = "builtin_tool_end"


class BuiltinToolResult(BaseEvent):
    """Provider returned a result for a built-in tool call."""

    result: messages.BuiltinToolReturnPart
    tool_call_id: str = ""

    kind: Literal["builtin_tool_result"] = "builtin_tool_result"


class FileEvent(BaseEvent):
    """A complete generated file from the LLM."""

    block_id: str = ""
    media_type: str
    data: str | bytes
    filename: str | None = None

    kind: Literal["file"] = "file"


class HookSuspension(BaseEvent):
    kind: Literal["hook_suspension"] = "hook_suspension"


class HookResolution(BaseEvent):
    kind: Literal["hook_resolution"] = "hook_resolution"


Event = (
    StreamStart
    | StreamEnd
    | TextStart
    | TextDelta
    | TextEnd
    | ReasoningStart
    | ReasoningDelta
    | ReasoningEnd
    | ToolStart
    | ToolDelta
    | ToolEnd
    | BuiltinToolStart
    | BuiltinToolDelta
    | BuiltinToolEnd
    | BuiltinToolResult
    | FileEvent
    | HookSuspension
    | HookResolution
)

DiscriminatedEvent = Annotated[
    Event,
    pydantic.Field(discriminator="kind"),
]


async def replay_message_events(
    msg: messages.Message,
) -> AsyncGenerator[Event]:
    """Synthesize stream events for ``msg``.

    Use when you have a complete ``Message`` from a non-streaming source —
    e.g., the result of a Temporal activity, a cached LLM response, or an
    offline test fixture — and want to feed it through code that consumes
    an async event stream (``ai.Stream``, ``ai.ToolRunner``, custom loops
    that mirror the default loop's shape, etc.)::

        async with ai.Stream(ai.events.replay_message_events(msg)) as stream:
            async with ai.ToolRunner() as tr:
                async for event in ai.util.merge(stream, tr.events()):
                    ...

    Each part is emitted as the start/delta/end triple a streaming adapter
    would have produced, in part order, bracketed by ``StreamStart`` and
    ``StreamEnd``. The full body of text/reasoning/tool-args is sent as a
    single delta — the granularity of the model's original chunking is
    not recoverable from a complete message.

    Parts with no model-layer event analog — ``ToolResultPart``,
    ``HookPart`` — are skipped silently; they are agent-layer concerns
    and never appear on the model stream.
    """
    yield StreamStart()
    for part in msg.parts:
        if isinstance(part, messages.TextPart):
            yield TextStart(block_id=part.id)
            if part.text:
                yield TextDelta(block_id=part.id, chunk=part.text)
            yield TextEnd(block_id=part.id)
        elif isinstance(part, messages.ReasoningPart):
            yield ReasoningStart(block_id=part.id)
            if part.text:
                yield ReasoningDelta(block_id=part.id, chunk=part.text)
            yield ReasoningEnd(block_id=part.id)
        elif isinstance(part, messages.ToolCallPart):
            yield ToolStart(
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
            )
            if part.tool_args:
                yield ToolDelta(
                    tool_call_id=part.tool_call_id,
                    chunk=part.tool_args,
                )
            yield ToolEnd(tool_call_id=part.tool_call_id, tool_call=part)
        elif isinstance(part, messages.BuiltinToolCallPart):
            yield BuiltinToolStart(
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
            )
            if part.tool_args:
                yield BuiltinToolDelta(
                    tool_call_id=part.tool_call_id,
                    chunk=part.tool_args,
                )
            yield BuiltinToolEnd(tool_call_id=part.tool_call_id, tool_call=part)
        elif isinstance(part, messages.BuiltinToolReturnPart):
            yield BuiltinToolResult(tool_call_id=part.tool_call_id, result=part)
        elif isinstance(part, messages.FilePart):
            yield FileEvent(
                block_id=part.id,
                data=part.data,
                media_type=part.media_type,
                filename=part.filename,
            )
    yield StreamEnd()


# ---------------------------------------------------------------------------
# Agent-layer event types
#
# These extend the model-streaming ``Event`` vocabulary with events that
# originate in the agent runtime: tool-execution outcomes and hook
# suspension points.
# ---------------------------------------------------------------------------


class Aggregator[Item, Result, ModelInput]:
    @abc.abstractmethod
    def feed(self, item: Item) -> None: ...

    @abc.abstractmethod
    def snapshot(self) -> Result: ...

    def get_model_input(self) -> ModelInput:
        """Return the model-facing value derived from this aggregator's state.

        Default implementation defers to :meth:`to_model_input`; subclasses
        with non-trivial state may override either or both.
        """
        return type(self).to_model_input(self.snapshot())

    @classmethod
    @abc.abstractmethod
    def to_model_input(cls, snapshot: Result) -> ModelInput:
        """Stateless conversion: snapshot -> model-facing value.

        Called on inbound (when a tool result round-trips back from the
        wire) and anywhere else a snapshot needs to be re-derived
        without a live aggregator instance.
        """
        ...


class PartialToolCallResult(pydantic.BaseModel):
    """Emitted when tool calls or other yield_from callers yield values."""

    tool_call_id: str | None = None
    tool_name: str | None = None
    label: object = None
    value: Any = None

    def key(self) -> object:
        return (self.tool_call_id, self.label)

    aggregator_factory: Callable[[], Aggregator[Any, Any, Any]] | None = (
        pydantic.Field(default=None, exclude=True, repr=False)
    )

    kind: Literal["partial_tool_call_result"] = "partial_tool_call_result"


class ToolCallResult(pydantic.BaseModel):
    """Emitted after tool calls execute — carries the result message.

    When the framework auto-catches an exception raised by the tool,
    ``exception`` carries the real ``BaseException`` (with traceback /
    ``__cause__`` intact) so loops can log it richly.  The wire-bound
    ``ToolResultPart.result`` still has ``str(exc)`` for the LLM.
    The ``exception`` field is excluded from serialization.
    """

    message: messages.Message
    results: Sequence[messages.ToolResultPart]
    exception: BaseException | None = pydantic.Field(
        default=None, exclude=True, repr=False
    )

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["tool_call_result"] = "tool_call_result"


class HookEvent(pydantic.BaseModel):
    """Emitted when a hook suspends, resolves, or is cancelled."""

    message: messages.Message
    hook: messages.HookPart[Any]

    kind: Literal["hook"] = "hook"


AgentMessageEvent = Event | ToolCallResult | HookEvent

AgentEvent = Event | ToolCallResult | HookEvent | PartialToolCallResult

TerminalEvent = StreamEnd | ToolCallResult | HookEvent
