"""Live event stream conversion for the AI SDK UI protocol."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import pydantic

from ....types import events as events_
from ....types import media
from ....types import messages as messages_
from ...agent import MessageBundle
from . import approvals, outbound_messages, ui_events
from .tool_utils import normalize_tool_input

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable


def _tool_error_text(part: messages_.ToolResultPart) -> str:
    """Best-effort error text extraction from a failed tool result."""
    if isinstance(part.result, str) and part.result:
        return part.result
    if isinstance(part.result, dict):
        for key in ("error", "message", "detail"):
            value = part.result.get(key)
            if isinstance(value, str) and value:
                return value
    return "Tool execution failed"


def _to_wire_output(snapshot: Any) -> Any:
    """Convert an aggregator snapshot to its UI wire representation.

    For ``MessageBundle`` (sub-agent transcripts) this produces a single
    ``UIMessage`` assistant bubble — the canonical AI SDK shape.  Other
    snapshot types pass through unchanged.

    Returns ``None`` if the bundle has no assistant anchor yet (e.g. a
    streaming sub-agent that has produced no messages); callers should
    skip emitting in that case.
    """
    if isinstance(snapshot, MessageBundle):
        ui_msgs = outbound_messages.to_ui_messages(list(snapshot.messages))
        return ui_msgs[-1] if ui_msgs else None
    return snapshot


class _StreamState:
    """Single-pass state across one ``to_stream()`` call."""

    def __init__(self) -> None:
        self.ui_message_id: str | None = None
        self.emitted_start: bool = False
        self.in_step: bool = False

        self.started_tool_inputs: set[str] = set()
        self.tool_names: dict[str, str] = {}
        self.input_available_emitted: set[str] = set()
        self.emitted_tool_results: set[str] = set()
        self.emitted_approval_requests: set[str] = set()

        self.open_text_ids: set[str] = set()
        self.open_reasoning_ids: set[str] = set()
        self.completed_text_ids: set[str] = set()
        self.completed_reasoning_ids: set[str] = set()
        self.text_delta_ids: set[str] = set()
        self.reasoning_delta_ids: set[str] = set()

        # Per-tool-call aggregators for streaming generator tools.  Each
        # PartialToolCallResult feeds its value into the aggregator and
        # the snapshot goes out as a preliminary tool output.
        self.partial_aggregators: dict[
            str, events_.Aggregator[Any, Any, Any]
        ] = {}

    # -- boundary helpers ----------------------------------------------------

    def _close_open_blocks(self) -> list[ui_events.UIMessageStreamEvent]:
        events: list[ui_events.UIMessageStreamEvent] = []
        for rid in list(self.open_reasoning_ids):
            events.append(ui_events.UIReasoningEndEvent(id=rid))
            self.completed_reasoning_ids.add(rid)
        self.open_reasoning_ids.clear()
        for tid in list(self.open_text_ids):
            events.append(ui_events.UITextEndEvent(id=tid))
            self.completed_text_ids.add(tid)
        self.open_text_ids.clear()
        return events

    def _ensure_started(
        self,
        message_id: str | None = None,
    ) -> list[ui_events.UIMessageStreamEvent]:
        """Lazily emit UIStartEvent / UIStartStepEvent on the first event."""
        events: list[ui_events.UIMessageStreamEvent] = []

        if not self.emitted_start:
            self.ui_message_id = message_id
            events.append(ui_events.UIStartEvent(message_id=self.ui_message_id))
            events.append(ui_events.UIStartStepEvent())
            self.emitted_start = True
            self.in_step = True
            self.started_tool_inputs.clear()
            self.tool_names.clear()
            self.input_available_emitted.clear()
            self.emitted_tool_results.clear()
            self.emitted_approval_requests.clear()

        return events

    # -- phase: streaming events --------------------------------------------

    def on_event(
        self, event: events_.Event
    ) -> list[ui_events.UIMessageStreamEvent]:
        out: list[ui_events.UIMessageStreamEvent] = []

        # Lazily open the UI message on the first streaming event.
        if not self.emitted_start:
            message = event.message
            message_id = None
            if message.role == "assistant":
                if message.turn_id is not None:
                    message_id = message.turn_id
                elif message.id != "<unset>":
                    message_id = message.id
            out.extend(self._ensure_started(message_id))

        match event:
            case events_.TextStart(block_id=pid):
                self.open_text_ids.add(pid)
                out.append(
                    ui_events.UITextStartEvent(
                        id=pid,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.TextDelta(block_id=pid, chunk=chunk):
                if pid not in self.open_text_ids:
                    self.open_text_ids.add(pid)
                    out.append(
                        ui_events.UITextStartEvent(
                            id=pid,
                            provider_metadata=event.provider_metadata,
                        )
                    )
                self.text_delta_ids.add(pid)
                out.append(
                    ui_events.UITextDeltaEvent(
                        id=pid,
                        delta=chunk,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.TextEnd(block_id=pid):
                if pid in self.open_text_ids:
                    self.open_text_ids.discard(pid)
                    self.completed_text_ids.add(pid)
                    out.append(
                        ui_events.UITextEndEvent(
                            id=pid,
                            provider_metadata=event.provider_metadata,
                        )
                    )

            case events_.ReasoningStart(block_id=pid):
                self.open_reasoning_ids.add(pid)
                out.append(
                    ui_events.UIReasoningStartEvent(
                        id=pid,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.ReasoningDelta(block_id=pid, chunk=chunk):
                if pid not in self.open_reasoning_ids:
                    self.open_reasoning_ids.add(pid)
                    out.append(
                        ui_events.UIReasoningStartEvent(
                            id=pid,
                            provider_metadata=event.provider_metadata,
                        )
                    )
                self.reasoning_delta_ids.add(pid)
                out.append(
                    ui_events.UIReasoningDeltaEvent(
                        id=pid,
                        delta=chunk,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.ReasoningEnd(block_id=pid):
                if pid in self.open_reasoning_ids:
                    self.open_reasoning_ids.discard(pid)
                    self.completed_reasoning_ids.add(pid)
                    out.append(
                        ui_events.UIReasoningEndEvent(
                            id=pid,
                            provider_metadata=event.provider_metadata,
                        )
                    )

            case events_.ToolStart(tool_call_id=tcid, tool_name=name):
                self.tool_names[tcid] = name
                if tcid in self.started_tool_inputs:
                    return out
                self.started_tool_inputs.add(tcid)
                out.append(
                    ui_events.UIToolInputStartEvent(
                        tool_call_id=tcid,
                        tool_name=name,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.ToolDelta(tool_call_id=tcid, chunk=chunk):
                if tcid not in self.started_tool_inputs:
                    self.started_tool_inputs.add(tcid)
                    out.append(
                        ui_events.UIToolInputStartEvent(
                            tool_call_id=tcid,
                            tool_name=self.tool_names.get(tcid, ""),
                            provider_metadata=event.provider_metadata,
                        )
                    )
                out.append(
                    ui_events.UIToolInputDeltaEvent(
                        tool_call_id=tcid,
                        input_text_delta=chunk,
                    )
                )

            case events_.ToolEnd():
                pass

            case events_.BuiltinToolStart(tool_call_id=tcid, tool_name=name):
                self.tool_names[tcid] = name
                if tcid in self.started_tool_inputs:
                    return out
                self.started_tool_inputs.add(tcid)
                out.append(
                    ui_events.UIToolInputStartEvent(
                        tool_call_id=tcid,
                        tool_name=name,
                        provider_executed=True,
                        provider_metadata=event.provider_metadata,
                        dynamic=True,
                    )
                )

            case events_.BuiltinToolDelta(tool_call_id=tcid, chunk=chunk):
                if tcid not in self.started_tool_inputs:
                    self.started_tool_inputs.add(tcid)
                    out.append(
                        ui_events.UIToolInputStartEvent(
                            tool_call_id=tcid,
                            tool_name=self.tool_names.get(tcid, ""),
                            provider_executed=True,
                            provider_metadata=event.provider_metadata,
                            dynamic=True,
                        )
                    )
                out.append(
                    ui_events.UIToolInputDeltaEvent(
                        tool_call_id=tcid,
                        input_text_delta=chunk,
                    )
                )

            case events_.BuiltinToolEnd(tool_call_id=tcid, tool_call=tc):
                if tcid not in self.input_available_emitted:
                    self.input_available_emitted.add(tcid)
                    out.append(
                        ui_events.UIToolInputAvailableEvent(
                            tool_call_id=tcid,
                            tool_name=tc.tool_name,
                            input=normalize_tool_input(tc.tool_args),
                            provider_executed=True,
                            provider_metadata=tc.provider_metadata
                            or event.provider_metadata,
                            dynamic=True,
                        )
                    )

            case events_.BuiltinToolResult(tool_call_id=tcid, result=result):
                if tcid in self.emitted_tool_results:
                    return out
                self.emitted_tool_results.add(tcid)
                if result.is_error:
                    out.append(
                        ui_events.UIToolOutputErrorEvent(
                            tool_call_id=tcid,
                            error_text=str(result.result),
                            provider_executed=True,
                            provider_metadata=result.provider_metadata
                            or event.provider_metadata,
                            dynamic=True,
                        )
                    )
                else:
                    out.append(
                        ui_events.UIToolOutputAvailableEvent(
                            tool_call_id=tcid,
                            output=result.result,
                            provider_executed=True,
                            provider_metadata=result.provider_metadata
                            or event.provider_metadata,
                            dynamic=True,
                        )
                    )

            case events_.FileEvent(
                media_type=media_type,
                data=data,
            ):
                out.append(
                    ui_events.UIFileEvent(
                        url=media.data_to_data_url(data, media_type),
                        media_type=media_type,
                        provider_metadata=event.provider_metadata,
                    )
                )

        return out

    # -- phase: tool results ------------------------------------------------

    def on_tool_result(
        self, event: events_.ToolCallResult
    ) -> list[ui_events.UIMessageStreamEvent]:
        """Handle a ``ToolCallResult`` — emit tool input/output events."""
        msg = event.message
        out: list[ui_events.UIMessageStreamEvent] = []

        out.extend(self._ensure_started(msg.turn_id))

        # Emit ToolInputAvailable for each tool call that triggered
        # these results (from the assistant message's ToolCallParts).
        for part in msg.parts:
            if isinstance(part, messages_.ToolCallPart):
                if part.tool_call_id in self.input_available_emitted:
                    continue
                self.input_available_emitted.add(part.tool_call_id)
                if part.tool_call_id not in self.started_tool_inputs:
                    self.started_tool_inputs.add(part.tool_call_id)
                    out.append(
                        ui_events.UIToolInputStartEvent(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            provider_metadata=part.provider_metadata,
                        )
                    )
                out.append(
                    ui_events.UIToolInputAvailableEvent(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        input=normalize_tool_input(part.tool_args),
                        provider_metadata=part.provider_metadata,
                    )
                )

        # Emit tool results.
        for part in event.results:
            if part.tool_call_id in self.emitted_tool_results:
                continue
            # Hook-abort placeholders are internal bookkeeping: the
            # corresponding HookPart(pending) drives the UI state.
            if part.is_hook_pending:
                continue
            self.emitted_tool_results.add(part.tool_call_id)
            if part.is_error:
                out.append(
                    ui_events.UIToolOutputErrorEvent(
                        tool_call_id=part.tool_call_id,
                        error_text=_tool_error_text(part),
                        provider_metadata=part.provider_metadata,
                    )
                )
            else:
                wire_output = _to_wire_output(part.result)
                if wire_output is None:
                    # Aggregator produced no anchor (e.g. sub-agent
                    # tool that yielded nothing).  Skip the final
                    # output emit; preliminaries already covered the
                    # streaming view if any.
                    continue
                out.append(
                    ui_events.UIToolOutputAvailableEvent(
                        tool_call_id=part.tool_call_id,
                        output=wire_output,
                        provider_metadata=part.provider_metadata,
                    )
                )

        return out

    def on_partial_tool_result(
        self, event: events_.PartialToolCallResult
    ) -> list[ui_events.UIMessageStreamEvent]:
        """Feed the value and emit a preliminary output.

        Each PartialToolCallResult carries one yielded value plus the
        aggregator factory the tool was declared with.  We instantiate
        the aggregator once per ``tool_call_id`` and use its snapshot
        as the ``output`` of a preliminary ``UIToolOutputAvailableEvent``.
        The AI SDK supersedes preliminary outputs with the final
        ``ToolCallResult`` when it arrives.
        """
        out: list[ui_events.UIMessageStreamEvent] = []

        tcid = event.tool_call_id
        factory = event.aggregator_factory
        if tcid is None or factory is None:
            return out

        out.extend(self._ensure_started())

        agg = self.partial_aggregators.get(tcid)
        if agg is None:
            agg = factory()
            self.partial_aggregators[tcid] = agg
        agg.feed(event.value)

        wire_output = _to_wire_output(agg.snapshot())
        if wire_output is None:
            # Sub-agent bundle without an assistant anchor yet — wait
            # for more events before emitting.
            return out

        out.append(
            ui_events.UIToolOutputAvailableEvent(
                tool_call_id=tcid,
                output=wire_output,
                preliminary=True,
            )
        )
        return out

    # -- phase: hooks -------------------------------------------------------

    def on_hook(
        self, event: events_.HookEvent
    ) -> list[ui_events.UIMessageStreamEvent]:
        """Handle a ``HookEvent`` — emit approval events."""
        hook_part = event.hook
        out: list[ui_events.UIMessageStreamEvent] = []

        # Ensure the UI message is started.
        out.extend(self._ensure_started(event.message.turn_id))

        tc_id = approvals.tool_call_id_for(hook_part)
        if tc_id is None:
            return out

        is_automatic = hook_part.metadata.get("isAutomatic")
        is_automatic = is_automatic if isinstance(is_automatic, bool) else None
        match hook_part.status:
            case "pending":
                if tc_id in self.emitted_approval_requests:
                    return out
                self.emitted_approval_requests.add(tc_id)
                out.append(
                    ui_events.UIToolApprovalRequestEvent(
                        approval_id=hook_part.hook_id,
                        tool_call_id=tc_id,
                        is_automatic=is_automatic,
                    )
                )
            case "resolved":
                resolution: dict[str, Any] = hook_part.resolution or {}
                provider_executed = hook_part.metadata.get("providerExecuted")
                provider_executed = (
                    provider_executed
                    if isinstance(provider_executed, bool)
                    else None
                )
                provider_metadata = hook_part.metadata.get(
                    "callProviderMetadata"
                )
                provider_metadata = (
                    provider_metadata
                    if isinstance(provider_metadata, dict)
                    else None
                )
                out.append(
                    ui_events.UIToolApprovalResponseEvent(
                        approval_id=hook_part.hook_id,
                        approved=bool(resolution.get("granted")),
                        reason=resolution.get("reason"),
                        provider_executed=provider_executed,
                        provider_metadata=provider_metadata,
                    )
                )
                if not resolution.get("granted"):
                    out.append(
                        ui_events.UIToolOutputDeniedEvent(tool_call_id=tc_id)
                    )
            case "cancelled":
                out.append(
                    ui_events.UIToolOutputErrorEvent(
                        tool_call_id=tc_id,
                        error_text="Hook cancelled",
                    )
                )

        return out

    # -- phase: stream finish ------------------------------------------------

    def finish(self) -> list[ui_events.UIMessageStreamEvent]:
        events = self._close_open_blocks()
        if self.in_step:
            events.append(ui_events.UIFinishStepEvent())
            self.in_step = False
        if self.emitted_start:
            events.append(ui_events.UIFinishEvent(finish_reason="stop"))
        return events


async def to_stream(
    events: AsyncIterable[events_.AgentEvent],
) -> AsyncGenerator[ui_events.UIMessageStreamEvent]:
    """Walk internal events once, emitting AI SDK UI stream events."""
    state = _StreamState()

    async for event in events:
        match event:
            case events_.ToolCallResult():
                for ui_event in state.on_tool_result(event):
                    yield ui_event
            case events_.PartialToolCallResult():
                for ui_event in state.on_partial_tool_result(event):
                    yield ui_event
            case events_.HookEvent():
                for ui_event in state.on_hook(event):
                    yield ui_event
            case _:
                for ui_event in state.on_event(event):
                    yield ui_event

    for ui_event in state.finish():
        yield ui_event


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pydantic.BaseModel):
        return obj.model_dump(mode="json", by_alias=True)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def serialize_event(event: ui_events.UIMessageStreamEvent) -> str:
    """Serialize a stream event to JSON with camelCase keys."""
    d = dataclasses.asdict(event)
    if isinstance(event, ui_events.UIDataEvent):
        d["type"] = event.type
        del d["data_type"]
    camel_dict = {_to_camel_case(k): v for k, v in d.items() if v is not None}
    return json.dumps(camel_dict, default=_json_default)


def format_sse(event: ui_events.UIMessageStreamEvent) -> str:
    """Format a stream event as an SSE data line."""
    return f"data: {serialize_event(event)}\n\n"


def format_done_sse() -> str:
    """Format the AI SDK UI stream termination marker."""
    return "data: [DONE]\n\n"


async def to_sse(
    events: AsyncIterable[events_.AgentEvent],
) -> AsyncGenerator[str]:
    """Convert an internal event stream into SSE strings."""
    async for event in to_stream(events):
        yield format_sse(event)
    yield format_done_sse()
