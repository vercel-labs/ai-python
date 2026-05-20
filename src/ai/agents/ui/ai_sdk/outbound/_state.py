"""Stream state bookkeeping for the event-first outbound walk."""

from __future__ import annotations

import json
from typing import Any

from .....types import events as events_
from .....types import media
from .....types import messages as messages_
from ....agent import MessageBundle
from .. import _approvals, ui_events
from . import history


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


def _normalize_tool_input(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _metadata_bool(metadata: dict[str, Any], key: str) -> bool | None:
    value = metadata.get(key)
    return value if isinstance(value, bool) else None


def _metadata_dict(
    metadata: dict[str, Any],
    key: str,
) -> dict[str, Any] | None:
    value = metadata.get(key)
    return value if isinstance(value, dict) else None


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
        ui_msgs = history.to_ui_messages(list(snapshot.messages))
        return ui_msgs[-1] if ui_msgs else None
    return snapshot


def _stream_message_id(event: events_.Event) -> str | None:
    message = event.message
    if message.role != "assistant":
        return None
    if message.turn_id is not None:
        return message.turn_id
    return None if message.id == "<unset>" else message.id


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

    def _close_open_blocks(self) -> list[ui_events.UIMessageStreamPart]:
        parts: list[ui_events.UIMessageStreamPart] = []
        for rid in list(self.open_reasoning_ids):
            parts.append(ui_events.ReasoningEndPart(id=rid))
            self.completed_reasoning_ids.add(rid)
        self.open_reasoning_ids.clear()
        for tid in list(self.open_text_ids):
            parts.append(ui_events.TextEndPart(id=tid))
            self.completed_text_ids.add(tid)
        self.open_text_ids.clear()
        return parts

    def _finish_step(self) -> list[ui_events.UIMessageStreamPart]:
        parts = self._close_open_blocks()
        if self.in_step:
            parts.append(ui_events.FinishStepPart())
            self.in_step = False
        return parts

    def _reset_step_tracking(self) -> None:
        self.started_tool_inputs.clear()
        self.tool_names.clear()
        self.input_available_emitted.clear()
        self.emitted_tool_results.clear()
        self.emitted_approval_requests.clear()

    def _ensure_started(
        self,
        message_id: str | None = None,
    ) -> list[ui_events.UIMessageStreamPart]:
        """Lazily emit StartPart / StartStepPart on the first event."""
        parts: list[ui_events.UIMessageStreamPart] = []

        if not self.emitted_start:
            self.ui_message_id = message_id
            parts.append(ui_events.StartPart(message_id=self.ui_message_id))
            parts.append(ui_events.StartStepPart())
            self.emitted_start = True
            self.in_step = True
            self._reset_step_tracking()

        return parts

    # -- phase: streaming events --------------------------------------------

    def on_event(
        self, event: events_.Event
    ) -> list[ui_events.UIMessageStreamPart]:
        out: list[ui_events.UIMessageStreamPart] = []

        # Lazily open the UI message on the first streaming event.
        if not self.emitted_start:
            out.extend(self._ensure_started(_stream_message_id(event)))

        match event:
            case events_.TextStart(block_id=pid):
                self.open_text_ids.add(pid)
                out.append(
                    ui_events.TextStartPart(
                        id=pid,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.TextDelta(block_id=pid, chunk=chunk):
                if pid not in self.open_text_ids:
                    self.open_text_ids.add(pid)
                    out.append(
                        ui_events.TextStartPart(
                            id=pid,
                            provider_metadata=event.provider_metadata,
                        )
                    )
                self.text_delta_ids.add(pid)
                out.append(
                    ui_events.TextDeltaPart(
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
                        ui_events.TextEndPart(
                            id=pid,
                            provider_metadata=event.provider_metadata,
                        )
                    )

            case events_.ReasoningStart(block_id=pid):
                self.open_reasoning_ids.add(pid)
                out.append(
                    ui_events.ReasoningStartPart(
                        id=pid,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.ReasoningDelta(block_id=pid, chunk=chunk):
                if pid not in self.open_reasoning_ids:
                    self.open_reasoning_ids.add(pid)
                    out.append(
                        ui_events.ReasoningStartPart(
                            id=pid,
                            provider_metadata=event.provider_metadata,
                        )
                    )
                self.reasoning_delta_ids.add(pid)
                out.append(
                    ui_events.ReasoningDeltaPart(
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
                        ui_events.ReasoningEndPart(
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
                    ui_events.ToolInputStartPart(
                        tool_call_id=tcid,
                        tool_name=name,
                        provider_metadata=event.provider_metadata,
                    )
                )

            case events_.ToolDelta(tool_call_id=tcid, chunk=chunk):
                if tcid not in self.started_tool_inputs:
                    self.started_tool_inputs.add(tcid)
                    out.append(
                        ui_events.ToolInputStartPart(
                            tool_call_id=tcid,
                            tool_name=self.tool_names.get(tcid, ""),
                            provider_metadata=event.provider_metadata,
                        )
                    )
                out.append(
                    ui_events.ToolInputDeltaPart(
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
                    ui_events.ToolInputStartPart(
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
                        ui_events.ToolInputStartPart(
                            tool_call_id=tcid,
                            tool_name=self.tool_names.get(tcid, ""),
                            provider_executed=True,
                            provider_metadata=event.provider_metadata,
                            dynamic=True,
                        )
                    )
                out.append(
                    ui_events.ToolInputDeltaPart(
                        tool_call_id=tcid,
                        input_text_delta=chunk,
                    )
                )

            case events_.BuiltinToolEnd(tool_call_id=tcid, tool_call=tc):
                if tcid not in self.input_available_emitted:
                    self.input_available_emitted.add(tcid)
                    out.append(
                        ui_events.ToolInputAvailablePart(
                            tool_call_id=tcid,
                            tool_name=tc.tool_name,
                            input=_normalize_tool_input(tc.tool_args),
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
                        ui_events.ToolOutputErrorPart(
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
                        ui_events.ToolOutputAvailablePart(
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
                    ui_events.FilePart(
                        url=media.data_to_data_url(data, media_type),
                        media_type=media_type,
                        provider_metadata=event.provider_metadata,
                    )
                )

        return out

    # -- phase: tool results ------------------------------------------------

    def on_tool_result(
        self, event: events_.ToolCallResult
    ) -> list[ui_events.UIMessageStreamPart]:
        """Handle a ``ToolCallResult`` — emit tool input/output parts."""
        msg = event.message
        out: list[ui_events.UIMessageStreamPart] = []

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
                        ui_events.ToolInputStartPart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            provider_metadata=part.provider_metadata,
                        )
                    )
                out.append(
                    ui_events.ToolInputAvailablePart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        input=_normalize_tool_input(part.tool_args),
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
                    ui_events.ToolOutputErrorPart(
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
                    ui_events.ToolOutputAvailablePart(
                        tool_call_id=part.tool_call_id,
                        output=wire_output,
                        provider_metadata=part.provider_metadata,
                    )
                )

        return out

    def on_partial_tool_result(
        self, event: events_.PartialToolCallResult
    ) -> list[ui_events.UIMessageStreamPart]:
        """Feed the value and emit a preliminary output.

        Each PartialToolCallResult carries one yielded value plus the
        aggregator factory the tool was declared with.  We instantiate
        the aggregator once per ``tool_call_id`` and use its snapshot
        as the ``output`` of a preliminary ``ToolOutputAvailablePart``.
        The AI SDK supersedes preliminary outputs with the final
        ``ToolCallResult`` when it arrives.
        """
        out: list[ui_events.UIMessageStreamPart] = []

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
            ui_events.ToolOutputAvailablePart(
                tool_call_id=tcid,
                output=wire_output,
                preliminary=True,
            )
        )
        return out

    # -- phase: hooks -------------------------------------------------------

    def on_hook(
        self, event: events_.HookEvent
    ) -> list[ui_events.UIMessageStreamPart]:
        """Handle a ``HookEvent`` — emit approval parts."""
        hook_part = event.hook
        out: list[ui_events.UIMessageStreamPart] = []

        # Ensure the UI message is started.
        out.extend(self._ensure_started(event.message.turn_id))

        tc_id = _approvals.tool_call_id_for(hook_part)
        if tc_id is None:
            return out

        if hook_part.status == "pending":
            if tc_id in self.emitted_approval_requests:
                return out
            self.emitted_approval_requests.add(tc_id)
            out.append(
                ui_events.ToolApprovalRequestPart(
                    approval_id=hook_part.hook_id,
                    tool_call_id=tc_id,
                    is_automatic=_metadata_bool(
                        hook_part.metadata, "isAutomatic"
                    ),
                )
            )
        elif hook_part.status == "resolved":
            resolution: dict[str, Any] = hook_part.resolution or {}
            out.append(
                ui_events.ToolApprovalResponsePart(
                    approval_id=hook_part.hook_id,
                    approved=bool(resolution.get("granted")),
                    reason=resolution.get("reason"),
                    provider_executed=_metadata_bool(
                        hook_part.metadata, "providerExecuted"
                    ),
                    provider_metadata=_metadata_dict(
                        hook_part.metadata, "callProviderMetadata"
                    ),
                )
            )
            if not resolution.get("granted"):
                out.append(ui_events.ToolOutputDeniedPart(tool_call_id=tc_id))
        elif hook_part.status == "cancelled":
            out.append(
                ui_events.ToolOutputErrorPart(
                    tool_call_id=tc_id,
                    error_text="Hook cancelled",
                )
            )

        return out

    # -- phase: stream finish ------------------------------------------------

    def finish(self) -> list[ui_events.UIMessageStreamPart]:
        parts = self._finish_step()
        if self.emitted_start:
            parts.append(ui_events.FinishPart(finish_reason="stop"))
        return parts
