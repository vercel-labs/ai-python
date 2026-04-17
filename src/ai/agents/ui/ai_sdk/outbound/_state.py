"""Stream state bookkeeping for the live outbound walk.

Owns message/step boundary logic (via ``turn_id`` + ``agent``), tracks
which parts have open text/reasoning blocks, and guards against
re-emission when the runtime re-yields an already-finalized message.
"""

from __future__ import annotations

from typing import Any

from .....types import messages as messages_
from .. import _approvals, protocol


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


class _StreamState:
    """Single-pass state across one ``to_stream()`` call."""

    def __init__(self) -> None:
        self.current_turn_id: str | None = None
        self.current_agent: str | None = None
        self.ui_message_id: str | None = None
        self.emitted_start: bool = False
        self.in_step: bool = False

        # Message-level dedup — an ``is_done`` message re-emitted as input to a
        # later ``stream()`` call must not fire events twice.
        self.seen_done: set[str] = set()

        # Tool-call dedup — keyed by tool_call_id.
        self.started_tool_inputs: set[str] = set()
        self.input_available_emitted: set[str] = set()
        self.emitted_tool_results: set[str] = set()
        self.emitted_approval_requests: set[str] = set()

        # Open streaming blocks — keyed by part id.
        self.open_text_ids: set[str] = set()
        self.open_reasoning_ids: set[str] = set()

    # -- boundary helpers ----------------------------------------------------

    def _close_open_blocks(self) -> list[protocol.UIMessageStreamPart]:
        parts: list[protocol.UIMessageStreamPart] = []
        for rid in list(self.open_reasoning_ids):
            parts.append(protocol.ReasoningEndPart(id=rid))
        self.open_reasoning_ids.clear()
        for tid in list(self.open_text_ids):
            parts.append(protocol.TextEndPart(id=tid))
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
        self.input_available_emitted.clear()
        self.emitted_tool_results.clear()
        self.emitted_approval_requests.clear()

    # -- phase: message start ------------------------------------------------

    def on_message(self, msg: messages_.Message) -> list[protocol.UIMessageStreamPart]:
        """Emit UIMessage/step boundary parts for *msg*."""
        parts: list[protocol.UIMessageStreamPart] = []

        agent_changed = (
            self.emitted_start
            and msg.agent is not None
            and msg.agent != self.current_agent
        )

        if not self.emitted_start or agent_changed:
            parts.extend(self._finish_step())
            if self.emitted_start:
                parts.append(protocol.FinishPart(finish_reason="stop"))

            self.ui_message_id = msg.id
            parts.append(protocol.StartPart(message_id=msg.id))
            parts.append(protocol.StartStepPart())
            self.emitted_start = True
            self.in_step = True
            self.current_agent = msg.agent
            self.current_turn_id = msg.turn_id
            self._reset_step_tracking()
            return parts

        # Same UIMessage — check for step boundary via turn_id change. Only
        # non-None → different-non-None transitions fire a step boundary;
        # None carries the current step (tool results yielded by the loop are
        # intentionally left unstamped until the next stream() stamps them).
        if (
            msg.turn_id is not None
            and self.current_turn_id is not None
            and msg.turn_id != self.current_turn_id
        ):
            parts.extend(self._finish_step())
            parts.append(protocol.StartStepPart())
            self.in_step = True
            self._reset_step_tracking()
            self.current_turn_id = msg.turn_id
        elif msg.turn_id is not None and self.current_turn_id is None:
            self.current_turn_id = msg.turn_id

        return parts

    # -- phase: per-event (mid-stream) ---------------------------------------

    def on_event(
        self,
        msg: messages_.Message,
        event: messages_.StreamEvent,
        parts_by_id: dict[str, messages_.Part],
    ) -> list[protocol.UIMessageStreamPart]:
        part = parts_by_id.get(event.part_id)
        if part is None:
            return []

        match event, part:
            case messages_.PartOpened(), messages_.TextPart():
                self.open_text_ids.add(event.part_id)
                return [protocol.TextStartPart(id=event.part_id)]

            case messages_.PartDelta(chunk=chunk), messages_.TextPart():
                if event.part_id not in self.open_text_ids:
                    self.open_text_ids.add(event.part_id)
                    return [
                        protocol.TextStartPart(id=event.part_id),
                        protocol.TextDeltaPart(id=event.part_id, delta=chunk),
                    ]
                return [protocol.TextDeltaPart(id=event.part_id, delta=chunk)]

            case messages_.PartClosed(), messages_.TextPart():
                if event.part_id in self.open_text_ids:
                    self.open_text_ids.discard(event.part_id)
                    return [protocol.TextEndPart(id=event.part_id)]
                return []

            case messages_.PartOpened(), messages_.ReasoningPart():
                self.open_reasoning_ids.add(event.part_id)
                return [protocol.ReasoningStartPart(id=event.part_id)]

            case messages_.PartDelta(chunk=chunk), messages_.ReasoningPart():
                if event.part_id not in self.open_reasoning_ids:
                    self.open_reasoning_ids.add(event.part_id)
                    return [
                        protocol.ReasoningStartPart(id=event.part_id),
                        protocol.ReasoningDeltaPart(id=event.part_id, delta=chunk),
                    ]
                return [protocol.ReasoningDeltaPart(id=event.part_id, delta=chunk)]

            case messages_.PartClosed(), messages_.ReasoningPart():
                if event.part_id in self.open_reasoning_ids:
                    self.open_reasoning_ids.discard(event.part_id)
                    return [protocol.ReasoningEndPart(id=event.part_id)]
                return []

            case messages_.PartOpened(), messages_.ToolCallPart() as tc:
                if tc.tool_call_id in self.started_tool_inputs:
                    return []
                self.started_tool_inputs.add(tc.tool_call_id)
                return [
                    protocol.ToolInputStartPart(
                        tool_call_id=tc.tool_call_id,
                        tool_name=tc.tool_name,
                    )
                ]

            case messages_.PartDelta(chunk=chunk), messages_.ToolCallPart() as tc:
                out: list[protocol.UIMessageStreamPart] = []
                if tc.tool_call_id not in self.started_tool_inputs:
                    self.started_tool_inputs.add(tc.tool_call_id)
                    out.append(
                        protocol.ToolInputStartPart(
                            tool_call_id=tc.tool_call_id,
                            tool_name=tc.tool_name,
                        )
                    )
                out.append(
                    protocol.ToolInputDeltaPart(
                        tool_call_id=tc.tool_call_id,
                        input_text_delta=chunk,
                    )
                )
                return out

            case messages_.PartClosed(), messages_.ToolCallPart():
                # ToolInputAvailablePart is emitted in ``on_terminal`` from
                # the terminal ``tool_args`` snapshot.
                return []

        return []

    # -- phase: terminal (tool results, approvals, final tool-input) ---------

    def on_terminal(self, msg: messages_.Message) -> list[protocol.UIMessageStreamPart]:
        if not msg.is_done:
            return []

        out: list[protocol.UIMessageStreamPart] = []

        # Close any blocks that were opened but didn't see an explicit
        # PartClosed (e.g. provider terminates abruptly — safety net).
        if msg.stream is not None:
            opened_ids = {
                e.part_id
                for e in msg.stream.events
                if isinstance(e, messages_.PartOpened)
            }
            for tid in list(self.open_text_ids):
                if tid in opened_ids and not any(
                    isinstance(e, messages_.PartClosed) and e.part_id == tid
                    for e in msg.stream.events
                ):
                    out.append(protocol.TextEndPart(id=tid))
                    self.open_text_ids.discard(tid)

        for part in msg.parts:
            if isinstance(part, messages_.ToolCallPart):
                if part.tool_call_id in self.input_available_emitted:
                    continue
                self.input_available_emitted.add(part.tool_call_id)
                # Ensure ToolInputStart was emitted (no streaming events case).
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

            elif isinstance(part, messages_.ToolResultPart):
                if part.tool_call_id in self.emitted_tool_results:
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
                    out.append(
                        protocol.ToolOutputAvailablePart(
                            tool_call_id=part.tool_call_id,
                            output=part.result,
                        )
                    )

            elif isinstance(part, messages_.HookPart):
                tc_id = _approvals.tool_call_id_for(part)
                if tc_id is None:
                    continue

                if part.status == "pending":
                    if tc_id in self.emitted_approval_requests:
                        continue
                    self.emitted_approval_requests.add(tc_id)
                    out.append(
                        protocol.ToolApprovalRequestPart(
                            approval_id=part.hook_id,
                            tool_call_id=tc_id,
                        )
                    )
                elif part.status == "resolved":
                    resolution: dict[str, Any] = part.resolution or {}
                    if not resolution.get("granted", False):
                        out.append(protocol.ToolOutputDeniedPart(tool_call_id=tc_id))
                elif part.status == "cancelled":
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
