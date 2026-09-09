import abc
import contextvars
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


# Pydantic doesn't let an Annotated serializer add to the serialization
# context, so OmitEventMessages uses a ContextVar instead.
_omit_event_messages: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ai_omit_event_messages", default=False
)


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


def _serialize_omitting_event_messages(
    value: Any,
    handler: pydantic.SerializerFunctionWrapHandler,
) -> Any:
    token = _omit_event_messages.set(True)
    try:
        return handler(value)
    finally:
        _omit_event_messages.reset(token)


type OmitEventMessages[T] = Annotated[
    T, pydantic.WrapSerializer(_serialize_omitting_event_messages)
]


def _validate_event_message(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"id"}:
        return {**value, "role": "assistant", "parts": []}
    return value


def _serialize_event_message(
    value: messages.Message,
    handler: pydantic.SerializerFunctionWrapHandler,
) -> Any:
    if _omit_event_messages.get():
        return {"id": value.id}
    return handler(value)


type _EventMessage = Annotated[
    messages.Message,
    pydantic.BeforeValidator(_validate_event_message),
    pydantic.WrapSerializer(_serialize_event_message),
]


class ModelEvent(BaseEvent):
    """Streamed out of a model request (``ai.stream``).

    ``message`` carries the in-progress (or final) assistant message; the
    streaming layer aggregates parts into it as deltas arrive and stamps
    a reference onto each yielded event (``Stream.__anext__``). ``usage``
    carries the latest usage value reported by the provider (latest-wins
    across the stream).
    """

    message: _EventMessage = _DUMMY_MESSAGE
    usage: usage_.Usage | None = None
    provider_metadata: dict[str, Any] | None = None


class StreamStart(ModelEvent):
    kind: Literal["stream_start"] = "stream_start"


class StreamEnd(ModelEvent):
    """End of a model response.

    ``finish_reason`` is why the model stopped.  The framework adopts
    the OpenTelemetry gen_ai finish-reason vocabulary as its own:
    ``stop``, ``length``, ``content_filter``, ``tool_call``, ``error``,
    plus ``other`` as a catch-all.  Provider adapters normalize their
    native stop reasons into it; a provider value with no equivalent
    becomes ``other`` with the raw value kept in ``provider_metadata``.

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


class MessageHydrator:
    """Reconstruct messages from a sequential agent event stream."""

    def __init__(self, seed_message: messages.Message | None = None) -> None:
        self._seed_message = seed_message
        self.message = seed_message or messages.Message(
            role="assistant", parts=[]
        )
        self.messages = [self.message]
        self.messages_by_id = {self.message.id: self.message}
        self._parts_by_message_id: dict[str, dict[str, messages.Part]] = {
            self.message.id: {}
        }
        self._message_selected = seed_message is not None
        self._seed_checked = False
        # A stream that exhausts without StreamEnd died mid-response.
        self.ended = False
        self.finish_reason: str | None = None
        self.response_id: str | None = None
        self.response_model: str | None = None

    def feed[T: BaseEvent](self, event: T) -> T:
        # ToolCallResult and HookEvent always carry full messages, so we just
        # use them.
        if isinstance(event, ToolCallResult | HookEvent):
            self.message = event.message
            existing = self.messages_by_id.get(event.message.id)
            if existing is None:
                self.messages.append(event.message)
            elif existing is not event.message:
                self.messages[self.messages.index(existing)] = event.message
            self.messages_by_id[event.message.id] = event.message
            return event
        if not isinstance(event, ModelEvent):
            return event

        updates: dict[str, Any] = {}
        message_id = event.message.id
        if message_id != "<unset>":
            if self._seed_message is not None and not self._seed_checked:
                self._seed_checked = True
                if message_id != self.message.id:
                    raise ValueError(
                        "seed message id does not match event message id: "
                        f"{self.message.id!r} != {message_id!r}"
                    )
            if message_id != self.message.id:
                self.ended = False
                if not self._message_selected:
                    old_id = self.message.id
                    del self.messages_by_id[old_id]
                    self.message.id = message_id
                    self.messages_by_id[message_id] = self.message
                    self._parts_by_message_id[message_id] = (
                        self._parts_by_message_id.pop(old_id)
                    )
                    self._message_selected = True
                else:
                    self.message = self.messages_by_id.get(
                        message_id
                    ) or messages.Message(
                        id=message_id, role="assistant", parts=[]
                    )
                    if message_id not in self.messages_by_id:
                        self.messages.append(self.message)
                        self.messages_by_id[message_id] = self.message
        self._parts = self._parts_by_message_id.setdefault(self.message.id, {})

        # Replay events carry no new state — the seeded message already
        # has everything they would have produced.  A replayed turn is
        # complete by construction, so it also counts as ended.
        if event.replay:
            self.ended = True
            return event.model_copy(update={"message": self.message})

        # grab usage from any event that carries one
        if event.usage is not None:
            self.message.usage = event.usage

        match event:
            case TextStart(block_id=bid, provider_metadata=pm):
                tp = messages.TextPart(id=bid, text="", provider_metadata=pm)
                self.message.parts.append(tp)
                self._parts[bid] = tp
            case TextDelta(block_id=bid, chunk=c, provider_metadata=pm):
                existing_text = self._parts.get(bid)
                if isinstance(existing_text, messages.TextPart):
                    existing_text.text += c
                    if pm is not None:
                        existing_text.provider_metadata = pm
            case TextEnd(block_id=bid, provider_metadata=pm):
                existing_text = self._parts.get(bid)
                if (
                    isinstance(existing_text, messages.TextPart)
                    and pm is not None
                ):
                    existing_text.provider_metadata = pm
            case ReasoningStart(block_id=bid, provider_metadata=pm):
                rp = messages.ReasoningPart(
                    id=bid, text="", provider_metadata=pm
                )
                self.message.parts.append(rp)
                self._parts[bid] = rp
            case ReasoningDelta(block_id=bid, chunk=c, provider_metadata=pm):
                existing_reasoning = self._parts.get(bid)
                if isinstance(existing_reasoning, messages.ReasoningPart):
                    existing_reasoning.text += c
                    if pm is not None:
                        existing_reasoning.provider_metadata = pm
            case ReasoningEnd(block_id=bid, provider_metadata=pm):
                existing_reasoning = self._parts.get(bid)
                if (
                    isinstance(existing_reasoning, messages.ReasoningPart)
                    and pm is not None
                ):
                    existing_reasoning.provider_metadata = pm
            case ToolStart(
                tool_call_id=tcid, tool_name=name, provider_metadata=pm
            ):
                tcp = messages.ToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args="",
                    provider_metadata=pm,
                )
                self.message.parts.append(tcp)
                self._parts[tcid] = tcp
            case ToolDelta(tool_call_id=tcid, chunk=c, provider_metadata=pm):
                existing_tool = self._parts.get(tcid)
                if isinstance(existing_tool, messages.ToolCallPart):
                    existing_tool.tool_args += c
                    if pm is not None:
                        existing_tool.provider_metadata = pm

            case ToolEnd(tool_call_id=tcid, provider_metadata=pm):
                existing_tool = self._parts.get(tcid)
                if isinstance(existing_tool, messages.ToolCallPart):
                    updates["tool_call"] = existing_tool
                    if pm is not None:
                        existing_tool.provider_metadata = pm
            case BuiltinToolStart(
                tool_call_id=tcid,
                tool_name=name,
                provider_metadata=pm,
            ):
                btcp = messages.BuiltinToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args="",
                    provider_metadata=pm,
                )
                self.message.parts.append(btcp)
                self._parts[tcid] = btcp
            case BuiltinToolDelta(
                tool_call_id=tcid, chunk=c, provider_metadata=pm
            ):
                existing_btc = self._parts.get(tcid)
                if isinstance(existing_btc, messages.BuiltinToolCallPart):
                    existing_btc.tool_args += c
                    if pm is not None:
                        existing_btc.provider_metadata = pm
            case BuiltinToolEnd(tool_call_id=tcid, provider_metadata=pm):
                existing_btc = self._parts.get(tcid)
                if isinstance(existing_btc, messages.BuiltinToolCallPart):
                    updates["tool_call"] = existing_btc
                    if pm is not None:
                        existing_btc.provider_metadata = pm
            case BuiltinToolResult(result=res, provider_metadata=pm):
                if pm is not None:
                    res = res.model_copy(update={"provider_metadata": pm})
                self.message.parts.append(res)
            case FileEvent(
                block_id=bid,
                media_type=mt,
                data=d,
                filename=fname,
                provider_metadata=pm,
            ):
                fp = messages.FilePart(
                    id=bid or messages.generate_id(),
                    data=d,
                    media_type=mt,
                    filename=fname,
                    provider_metadata=pm,
                )
                self.message.parts.append(fp)
                self._parts[fp.id] = fp

            case StreamEnd(
                provider_metadata=pm,
                finish_reason=finish,
                response_id=rid,
                response_model=rmodel,
            ):
                self.ended = True
                self.finish_reason = finish
                self.response_id = rid
                self.response_model = rmodel
                if pm is not None:
                    self.message.provider_metadata = pm
            case _:
                pass

        return event.model_copy(update={"message": self.message, **updates})


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
