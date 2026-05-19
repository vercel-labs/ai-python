"""Stream state bookkeeping for the event-first outbound walk."""

from __future__ import annotations

from typing import Any

from .....types import events as events_
from .....types import messages as messages_
from ....agent import MessageBundle
from .. import _approvals, protocol
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

    def _close_open_blocks(self) -> list[protocol.UIMessageStreamPart]:
        parts: list[protocol.UIMessageStreamPart] = []
        for rid in list(self.open_reasoning_ids):
            parts.append(protocol.ReasoningEndPart(id=rid))
            self.completed_reasoning_ids.add(rid)
        self.open_reasoning_ids.clear()
        for tid in list(self.open_text_ids):
            parts.append(protocol.TextEndPart(id=tid))
            self.completed_text_ids.add(tid)
        self.open_text_ids.clear()
        return parts

    def _finish_step(self) -> list[protocol.UIMessageStreamPart]:
        parts = self._close_open_blocks()
        if self.in_step:
            parts.append(protocol.FinishStepPart())
            self.in_step = False
        return parts

    def _reset_step_tracking(self) -> None:
        self.started_tool_inputs.clear()
        self.tool_names.clear()
        self.input_available_emitted.clear()
        self.emitted_tool_results.clear()
        self.emitted_approval_requests.clear()

    def _ensure_started(self) -> list[protocol.UIMessageStreamPart]:
        """Lazily emit StartPart / StartStepPart on the first event."""
        parts: list[protocol.UIMessageStreamPart] = []

        if not self.emitted_start:
            parts.append(protocol.StartPart(message_id=None))
            parts.append(protocol.StartStepPart())
            self.emitted_start = True
            self.in_step = True
            self._reset_step_tracking()

        return parts

    # -- phase: streaming events --------------------------------------------

    def on_event(
        self, event: events_.Event
    ) -> list[protocol.UIMessageStreamPart]:
        out: list[protocol.UIMessageStreamPart] = []

        # Lazily open the UI message on the first streaming event.
        if not self.emitted_start:
            out.extend(self._ensure_started())

        match event:
            case events_.TextStart(block_id=pid):
                self.open_text_ids.add(pid)
                out.append(protocol.TextStartPart(id=pid))

            case events_.TextDelta(block_id=pid, chunk=chunk):
                if pid not in self.open_text_ids:
                    self.open_text_ids.add(pid)
                    out.append(protocol.TextStartPart(id=pid))
                self.text_delta_ids.add(pid)
                out.append(protocol.TextDeltaPart(id=pid, delta=chunk))

            case events_.TextEnd(block_id=pid):
                if pid in self.open_text_ids:
                    self.open_text_ids.discard(pid)
                    self.completed_text_ids.add(pid)
                    out.append(protocol.TextEndPart(id=pid))

            case events_.ReasoningStart(block_id=pid):
                self.open_reasoning_ids.add(pid)
                out.append(protocol.ReasoningStartPart(id=pid))

            case events_.ReasoningDelta(block_id=pid, chunk=chunk):
                if pid not in self.open_reasoning_ids:
                    self.open_reasoning_ids.add(pid)
                    out.append(protocol.ReasoningStartPart(id=pid))
                self.reasoning_delta_ids.add(pid)
                out.append(protocol.ReasoningDeltaPart(id=pid, delta=chunk))

            case events_.ReasoningEnd(block_id=pid):
                if pid in self.open_reasoning_ids:
                    self.open_reasoning_ids.discard(pid)
                    self.completed_reasoning_ids.add(pid)
                    out.append(protocol.ReasoningEndPart(id=pid))

            case events_.ToolStart(tool_call_id=tcid, tool_name=name):
                self.tool_names[tcid] = name
                if tcid in self.started_tool_inputs:
                    return out
                self.started_tool_inputs.add(tcid)
                out.append(
                    protocol.ToolInputStartPart(
                        tool_call_id=tcid,
                        tool_name=name,
                    )
                )

            case events_.ToolDelta(tool_call_id=tcid, chunk=chunk):
                if tcid not in self.started_tool_inputs:
                    self.started_tool_inputs.add(tcid)
                    out.append(
                        protocol.ToolInputStartPart(
                            tool_call_id=tcid,
                            tool_name=self.tool_names.get(tcid, ""),
                        )
                    )
                out.append(
                    protocol.ToolInputDeltaPart(
                        tool_call_id=tcid,
                        input_text_delta=chunk,
                    )
                )

            case events_.ToolEnd():
                pass

        return out

    # -- phase: tool results ------------------------------------------------

    def on_tool_result(
        self, event: events_.ToolCallResult
    ) -> list[protocol.UIMessageStreamPart]:
        """Handle a ``ToolCallResult`` — emit tool input/output parts."""
        msg = event.message
        out: list[protocol.UIMessageStreamPart] = []

        out.extend(self._ensure_started())

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
                        protocol.ToolInputStartPart(
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                        )
                    )
                out.append(
                    protocol.ToolInputAvailablePart(
                        tool_call_id=part.tool_call_id,
                        tool_name=part.tool_name,
                        input=part.tool_args,
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
                    protocol.ToolOutputErrorPart(
                        tool_call_id=part.tool_call_id,
                        error_text=_tool_error_text(part),
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
                    protocol.ToolOutputAvailablePart(
                        tool_call_id=part.tool_call_id,
                        output=wire_output,
                    )
                )

        return out

    def on_partial_tool_result(
        self, event: events_.PartialToolCallResult
    ) -> list[protocol.UIMessageStreamPart]:
        """Feed the value and emit a preliminary output.

        Each PartialToolCallResult carries one yielded value plus the
        aggregator factory the tool was declared with.  We instantiate
        the aggregator once per ``tool_call_id`` and use its snapshot
        as the ``output`` of a preliminary ``ToolOutputAvailablePart``.
        The AI SDK supersedes preliminary outputs with the final
        ``ToolCallResult`` when it arrives.
        """
        out: list[protocol.UIMessageStreamPart] = []

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
            protocol.ToolOutputAvailablePart(
                tool_call_id=tcid,
                output=wire_output,
                preliminary=True,
            )
        )
        return out

    # -- phase: hooks -------------------------------------------------------

    def on_hook(
        self, event: events_.HookEvent
    ) -> list[protocol.UIMessageStreamPart]:
        """Handle a ``HookEvent`` — emit approval parts."""
        hook_part = event.hook
        out: list[protocol.UIMessageStreamPart] = []

        # Ensure the UI message is started.
        out.extend(self._ensure_started())

        tc_id = _approvals.tool_call_id_for(hook_part)
        if tc_id is None:
            return out

        if hook_part.status == "pending":
            if tc_id in self.emitted_approval_requests:
                return out
            self.emitted_approval_requests.add(tc_id)
            out.append(
                protocol.ToolApprovalRequestPart(
                    approval_id=hook_part.hook_id,
                    tool_call_id=tc_id,
                )
            )
        elif hook_part.status == "resolved":
            resolution: dict[str, Any] = hook_part.resolution or {}
            if not resolution.get("granted"):
                out.append(protocol.ToolOutputDeniedPart(tool_call_id=tc_id))
        elif hook_part.status == "cancelled":
            out.append(
                protocol.ToolOutputErrorPart(
                    tool_call_id=tc_id,
                    error_text="Hook cancelled",
                )
            )

        return out

    # -- phase: stream finish ------------------------------------------------

    def finish(self) -> list[protocol.UIMessageStreamPart]:
        parts = self._finish_step()
        if self.emitted_start:
            parts.append(protocol.FinishPart(finish_reason="stop"))
        return parts
