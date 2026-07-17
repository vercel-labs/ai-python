import abc
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Annotated, Any, Literal

import pydantic

from . import messages
from . import usage as usage_

# we're using pydantic because events are crossing
# serialization border in the case of durable execution


# Placeholder so ModelEvent.message is typed as Message (not Message | None).
# Stream.__anext__ stamps the real in-progress message before yielding,
# so consumers never see this value.
_DUMMY_MESSAGE = messages.Message(id="<unset>", role="assistant", parts=[])


class BaseEvent(pydantic.BaseModel):
    """Anything ``ai.stream`` or ``Agent.run`` yields.

    ``replay`` is set on synthetic events emitted when ``models.stream``
    short-circuits an existing assistant turn (resume-after-approval
    flows).  ``Agent.run`` drops replay-flagged events from the consumer-
    facing stream — the loop's tool dispatcher still consumes them
    internally.  Excluded from JSON: it's a control flag, not data.
    """

    replay: bool = pydantic.Field(default=False, exclude=True, repr=False)

    model_config = pydantic.ConfigDict(frozen=True)


class ModelEvent(BaseEvent):
    """Streamed out of a model request (``ai.stream``).

    ``message`` carries the in-progress (or final) assistant message; the
    streaming layer aggregates parts into it as deltas arrive and stamps
    a reference onto each yielded event (``Stream.__anext__``). ``usage``
    carries the latest usage value reported by the provider (latest-wins
    across the stream).
    """

    message: messages.Message = _DUMMY_MESSAGE
    usage: usage_.Usage | None = None
    provider_metadata: dict[str, Any] | None = None


class StreamStart(ModelEvent):
    kind: Literal["stream_start"] = "stream_start"


class StreamEnd(ModelEvent):
    """End of a model response.

    ``finish_reason`` is why the model stopped.  The framework adopts
    the OpenTelemetry gen_ai finish-reason vocabulary as its own:
    ``stop``, ``length``, ``content_filter``, ``tool_call``, ``error``.
    Provider adapters normalize their native stop reasons into it;
    provider values with no equivalent pass through verbatim.

    ``response_id``/``response_model`` identify the provider response —
    ``response_model`` can differ from the requested model under
    gateway routing or fallbacks.  All are ``None`` when the provider
    doesn't report them.
    """

    kind: Literal["stream_end"] = "stream_end"
    finish_reason: str | None = None
    response_id: str | None = None
    response_model: str | None = None


class TextStart(ModelEvent):
    block_id: str = ""

    kind: Literal["text_start"] = "text_start"


class TextDelta(ModelEvent):
    chunk: str
    block_id: str = ""

    kind: Literal["text_delta"] = "text_delta"


class TextEnd(ModelEvent):
    block_id: str = ""

    kind: Literal["text_end"] = "text_end"


class ReasoningStart(ModelEvent):
    block_id: str = ""

    kind: Literal["reasoning_start"] = "reasoning_start"


class ReasoningDelta(ModelEvent):
    chunk: str
    block_id: str = ""

    kind: Literal["reasoning_delta"] = "reasoning_delta"


class ReasoningEnd(ModelEvent):
    block_id: str = ""

    kind: Literal["reasoning_end"] = "reasoning_end"


class ToolStart(ModelEvent):
    tool_call_id: str = ""
    tool_name: str = ""

    kind: Literal["tool_start"] = "tool_start"


class ToolDelta(ModelEvent):
    chunk: str
    tool_call_id: str = ""

    kind: Literal["tool_delta"] = "tool_delta"


class ToolEnd(ModelEvent):
    tool_call: messages.ToolCallPart
    tool_call_id: str = ""

    kind: Literal["tool_end"] = "tool_end"


class BuiltinToolStart(ModelEvent):
    tool_call_id: str = ""
    tool_name: str = ""

    kind: Literal["builtin_tool_start"] = "builtin_tool_start"


class BuiltinToolDelta(ModelEvent):
    chunk: str
    tool_call_id: str = ""

    kind: Literal["builtin_tool_delta"] = "builtin_tool_delta"


class BuiltinToolEnd(ModelEvent):
    tool_call: messages.BuiltinToolCallPart
    tool_call_id: str = ""

    kind: Literal["builtin_tool_end"] = "builtin_tool_end"


class BuiltinToolResult(ModelEvent):
    """Provider returned a result for a built-in tool call."""

    result: messages.BuiltinToolReturnPart
    tool_call_id: str = ""

    kind: Literal["builtin_tool_result"] = "builtin_tool_result"


class FileEvent(ModelEvent):
    """A complete generated file from the LLM."""

    block_id: str = ""
    media_type: str
    data: str | bytes
    filename: str | None = None

    kind: Literal["file"] = "file"


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
)


async def _replay_message_events(
    msg: messages.Message,
) -> AsyncGenerator[Event]:
    """Synthesize stream events for ``msg``."""
    # See Stream.replay_message
    yield StreamStart()
    for part in msg.parts:
        if isinstance(part, messages.TextPart):
            yield TextStart(block_id=part.id)
            if part.text:
                yield TextDelta(block_id=part.id, chunk=part.text)
            yield TextEnd(
                block_id=part.id, provider_metadata=part.provider_metadata
            )
        elif isinstance(part, messages.ReasoningPart):
            yield ReasoningStart(block_id=part.id)
            if part.text:
                yield ReasoningDelta(block_id=part.id, chunk=part.text)
            yield ReasoningEnd(
                block_id=part.id,
                provider_metadata=part.provider_metadata,
            )
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
            yield ToolEnd(
                tool_call_id=part.tool_call_id,
                tool_call=part,
                provider_metadata=part.provider_metadata,
            )
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
            yield BuiltinToolEnd(
                tool_call_id=part.tool_call_id,
                tool_call=part,
                provider_metadata=part.provider_metadata,
            )
        elif isinstance(part, messages.BuiltinToolReturnPart):
            yield BuiltinToolResult(tool_call_id=part.tool_call_id, result=part)
        elif isinstance(part, messages.FilePart):
            yield FileEvent(
                block_id=part.id,
                data=part.data,
                media_type=part.media_type,
                filename=part.filename,
                provider_metadata=part.provider_metadata,
            )
    yield StreamEnd(provider_metadata=msg.provider_metadata)


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


class PartialToolCallResult(BaseEvent):
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


class ToolCallResult(BaseEvent):
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


class HookEvent(BaseEvent):
    """Emitted when a hook suspends, resolves, or is cancelled."""

    message: messages.Message
    hook: messages.HookPart[Any]

    kind: Literal["hook"] = "hook"


class RunBlocked(BaseEvent):
    """The run is blocked on hooks.

    Emitted when the run stops being able to make progress without
    external input: at least one hook is deferred, no model stream is
    producing events, and every in-flight tool call is suspended
    awaiting a hook.  Streaming consumers can use this to surface
    "waiting for approval" state without reconstructing it from
    tool/hook events.

    ``hooks`` is a snapshot of the deferred hooks the run is blocked on.

    There is no mirror "unblocked" event because it would be redundant:
    a blocked run can only resume via a hook resolution (or
    cancellation), so the next ``HookEvent`` with a non-``pending``
    status *is* the unblock signal.  Note the converse does not hold —
    a ``ToolCallResult`` carrying an ``is_hook_deferred`` placeholder
    (serverless abort) arrives while the run stays blocked, and the run
    then ends still blocked.
    """

    hooks: tuple[messages.HookPart[Any], ...] = ()

    kind: Literal["run_blocked"] = "run_blocked"


AgentEvent = Annotated[
    Event | ToolCallResult | HookEvent | PartialToolCallResult | RunBlocked,
    pydantic.Field(discriminator="kind"),
]


class RunStateTracker:
    """Fold an agent event stream into run state (blocked-on-hooks).

    A pure function of the event stream: feed every event in order and
    :meth:`feed` returns a :class:`RunBlocked` event whenever the run
    becomes blocked, else None (:attr:`blocked` flips back silently —
    see :class:`RunBlocked` for why no mirror event exists).  Works
    identically over a live run or a serialized replay of one.

    The fold reads three things:

    * hook state from :class:`HookEvent` (``pending`` adds, ``resolved``
      / ``cancelled`` removes);
    * model-stream activity from :class:`StreamStart` / :class:`StreamEnd`;
    * in-flight tool calls from the assistant message on
      :class:`StreamEnd` (scheduled) and :class:`ToolCallResult`
      (settled), matched by ``tool_call_id``.

    The run is blocked when at least one hook is deferred, no stream is
    producing, and every in-flight tool call is accounted for by a
    deferred hook's ``tool_call_id``.  Consequently the signal is only
    as good as the stream: loops must yield their ``StreamEnd`` (with
    the assistant message) for tool calls to be counted, and custom
    gating must pass ``tool_call_id=`` to ``ai.hook()`` — an
    unattributed hook while tools are in flight reads as "still busy"
    and suppresses the signal.
    """

    def __init__(self) -> None:
        self._deferred: dict[str, messages.HookPart[Any]] = {}
        self._in_flight: set[str] = set()
        self._streaming = 0
        self._blocked = False

    @property
    def blocked(self) -> bool:
        return self._blocked

    @property
    def deferred_hooks(self) -> list[messages.HookPart[Any]]:
        return list(self._deferred.values())

    def feed(self, event: AgentEvent) -> RunBlocked | None:
        match event:
            case StreamStart():
                self._streaming += 1
            case StreamEnd():
                # Loops may emit a bare StreamEnd without a StreamStart
                # (e.g. when the model was streamed out-of-band), so
                # clamp at zero.
                self._streaming = max(0, self._streaming - 1)
                self._in_flight.update(
                    tc.tool_call_id for tc in event.message.tool_calls
                )
            case ToolCallResult():
                self._in_flight.difference_update(
                    r.tool_call_id for r in event.results
                )
            case HookEvent():
                if event.hook.status == "pending":
                    self._deferred[event.hook.hook_id] = event.hook
                else:
                    self._deferred.pop(event.hook.hook_id, None)
            case _:
                return None

        attributed = {
            h.tool_call_id
            for h in self._deferred.values()
            if h.tool_call_id is not None
        }
        now = (
            bool(self._deferred)
            and not self._streaming
            and self._in_flight <= attributed
        )
        if now == self._blocked:
            return None
        self._blocked = now
        if not now:
            return None
        return RunBlocked(hooks=tuple(self._deferred.values()))
