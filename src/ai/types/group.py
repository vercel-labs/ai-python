import dataclasses

from ....types import messages as messages_


@dataclasses.dataclass
class TextStart:
    block_id: str


@dataclasses.dataclass
class TextDelta:
    block_id: str
    delta: str


@dataclasses.dataclass
class TextEnd:
    block_id: str


@dataclasses.dataclass
class ReasoningStart:
    block_id: str


@dataclasses.dataclass
class ReasoningDelta:
    block_id: str
    delta: str


@dataclasses.dataclass
class ReasoningEnd:
    block_id: str
    signature: str | None = None


@dataclasses.dataclass
class ToolStart:
    tool_call_id: str
    tool_name: str


@dataclasses.dataclass
class ToolArgsDelta:
    tool_call_id: str
    delta: str


@dataclasses.dataclass
class ToolEnd:
    tool_call_id: str


@dataclasses.dataclass
class FileEvent:
    """A complete generated file from the LLM (e.g. inline image from Gemini/GPT)."""

    block_id: str
    media_type: str
    data: str  # base64 string or data-URL from the gateway


@dataclasses.dataclass
class MessageDone:
    finish_reason: str | None = None
    usage: messages_.Usage | None = None


StreamEvent = (
    TextStart
    | TextDelta
    | TextEnd
    | ReasoningStart
    | ReasoningDelta
    | ReasoningEnd
    | ToolStart
    | ToolArgsDelta
    | ToolEnd
    | FileEvent
    | MessageDone
)


@dataclasses.dataclass
class StreamHandler:
    """
    Accumulates LLM adapter events and produces Messages with stateful parts.

    This is the normalization layer between LLM adapters and the rest of the system.
    Parts are tracked in a single ``_current_parts`` dict keyed by block/tool id,
    updated in place as events stream in.  Each event carries the just-constructed
    frozen part snapshot, so consumers never need to look parts up by id.
    """

    message_id: str = dataclasses.field(default_factory=messages_.generate_id)

    # Single source of truth for part state, keyed by id. Insertion order
    # preserves provider emission order.
    _current_parts: dict[str, messages_.Part] = dataclasses.field(default_factory=dict)

    # Active tracking
    _active_text_id: str | None = None
    _active_reasoning_id: str | None = None
    _active_tool_ids: set[str] = dataclasses.field(default_factory=set)

    _is_done: bool = False
    _usage: messages_.Usage | None = None

    def handle_event(self, event: StreamEvent) -> messages_.Message:
        """Process event and return current Message state."""

        # Sidecar events for this yield (reset each call).
        stream_events: list[messages_.StreamEvent] = []

        match event:
            case TextStart(block_id=bid):
                part: messages_.Part = messages_.TextPart(id=bid, text="")
                self._current_parts[bid] = part
                self._active_text_id = bid
                stream_events.append(messages_.PartOpened(part=part))

            case TextDelta(block_id=bid, delta=d):
                existing = self._current_parts[bid]
                assert isinstance(existing, messages_.TextPart)
                part = messages_.TextPart(id=bid, text=existing.text + d)
                self._current_parts[bid] = part
                stream_events.append(messages_.PartDelta(part=part, chunk=d))

            case TextEnd(block_id=bid):
                if self._active_text_id == bid:
                    self._active_text_id = None
                stream_events.append(
                    messages_.PartClosed(part=self._current_parts[bid])
                )

            case ReasoningStart(block_id=bid):
                part = messages_.ReasoningPart(id=bid, text="")
                self._current_parts[bid] = part
                self._active_reasoning_id = bid
                stream_events.append(messages_.PartOpened(part=part))

            case ReasoningDelta(block_id=bid, delta=d):
                existing = self._current_parts[bid]
                assert isinstance(existing, messages_.ReasoningPart)
                part = messages_.ReasoningPart(
                    id=bid,
                    text=existing.text + d,
                    signature=existing.signature,
                )
                self._current_parts[bid] = part
                stream_events.append(messages_.PartDelta(part=part, chunk=d))

            case ReasoningEnd(block_id=bid, signature=sig):
                existing = self._current_parts[bid]
                assert isinstance(existing, messages_.ReasoningPart)
                part = messages_.ReasoningPart(
                    id=bid, text=existing.text, signature=sig
                )
                self._current_parts[bid] = part
                if self._active_reasoning_id == bid:
                    self._active_reasoning_id = None
                stream_events.append(messages_.PartClosed(part=part))

            case ToolStart(tool_call_id=tcid, tool_name=name):
                part = messages_.ToolCallPart(
                    id=tcid,
                    tool_call_id=tcid,
                    tool_name=name,
                    tool_args="",
                )
                self._current_parts[tcid] = part
                self._active_tool_ids.add(tcid)
                stream_events.append(messages_.PartOpened(part=part))

            case ToolArgsDelta(tool_call_id=tcid, delta=d):
                existing = self._current_parts[tcid]
                assert isinstance(existing, messages_.ToolCallPart)
                part = messages_.ToolCallPart(
                    id=tcid,
                    tool_call_id=existing.tool_call_id,
                    tool_name=existing.tool_name,
                    tool_args=existing.tool_args + d,
                )
                self._current_parts[tcid] = part
                stream_events.append(messages_.PartDelta(part=part, chunk=d))

            case ToolEnd(tool_call_id=tcid):
                self._active_tool_ids.discard(tcid)
                stream_events.append(
                    messages_.PartClosed(part=self._current_parts[tcid])
                )

            case FileEvent(block_id=bid, media_type=mt, data=d):
                self._current_parts[bid] = messages_.FilePart(
                    id=bid, data=d, media_type=mt
                )

            case MessageDone(usage=usage):
                self._is_done = True
                self._usage = usage
                self._active_text_id = None
                self._active_reasoning_id = None
                self._active_tool_ids.clear()

        return self._build_message(stream_events)

    def _build_message(
        self,
        stream_events: list[messages_.StreamEvent],
    ) -> messages_.Message:
        return messages_.Message(
            id=self.message_id,
            role="assistant",
            parts=list(self._current_parts.values()),
            usage=self._usage if self._is_done else None,
            stream=messages_.StreamState(
                new_events=stream_events,
                is_done=self._is_done,
            ),
        )
