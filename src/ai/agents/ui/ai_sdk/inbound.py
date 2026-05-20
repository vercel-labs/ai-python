"""Inbound adapter from AI SDK v6 UIMessages to internal messages.

The primary entry point is :func:`to_messages`, which bundles normalization,
approval extraction, parsing, and pre-registration of approval resolutions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple

from ....types import messages as messages_
from ...agent import MessageBundle
from ...hooks import resolve_hook
from . import ui_message

logger = logging.getLogger(__name__)


_TOOL_RESULT_STATES: frozenset[str] = frozenset({"output-available"})
_TOOL_ERROR_STATES: frozenset[str] = frozenset(
    {"output-error", "output-denied"}
)


def _is_tool_completed(state: ui_message.UIToolInvocationState) -> bool:
    return state in _TOOL_RESULT_STATES or state in _TOOL_ERROR_STATES


def _is_tool_error(state: ui_message.UIToolInvocationState) -> bool:
    return state in _TOOL_ERROR_STATES


class _AdapterIdPolicy:
    """Stable ids for reconstructing canonical messages from UI bubbles."""

    @staticmethod
    def message_id(turn_id: str, role: str, index: int) -> str:
        return f"{turn_id}:{role}:{index}"

    @classmethod
    def with_stable_part_ids(
        cls,
        message_id: str,
        parts: list[messages_.Part],
    ) -> list[messages_.Part]:
        counts: dict[str, int] = {}
        result: list[messages_.Part] = []
        for index, part in enumerate(parts):
            base = cls.part_id(message_id, part, index)
            seen = counts.get(base, 0)
            counts[base] = seen + 1
            part_id = base if seen == 0 else f"{base}:{seen}"
            result.append(part.model_copy(update={"id": part_id}))
        return result

    @staticmethod
    def part_id(
        message_id: str,
        part: messages_.Part,
        index: int,
    ) -> str:
        match part:
            case messages_.TextPart():
                return f"{message_id}:text:{index}"
            case messages_.ReasoningPart():
                return f"{message_id}:reasoning:{index}"
            case messages_.ToolCallPart(tool_call_id=tool_call_id):
                return f"{message_id}:call:{tool_call_id}"
            case messages_.ToolResultPart(tool_call_id=tool_call_id):
                return f"{message_id}:result:{tool_call_id}"
            case messages_.BuiltinToolCallPart(tool_call_id=tool_call_id):
                return f"{message_id}:builtin-call:{tool_call_id}"
            case messages_.BuiltinToolReturnPart(tool_call_id=tool_call_id):
                return f"{message_id}:builtin-result:{tool_call_id}"
            case messages_.HookPart():
                return f"{message_id}:hook:{index}"
            case messages_.FilePart():
                return f"{message_id}:file:{index}"


# TODO(datamodel-rework §4): once tool args have a canonical shape, drop
# these normalizers.
def _normalize_tool_args(tool_input: Any) -> str:
    """Normalize tool input (JSON string, JSON value, or None) to a string."""
    match tool_input:
        case str():
            return tool_input
        case None:
            return "{}"
        case _:
            return json.dumps(tool_input)


def _tool_input_for_args(
    part: ui_message.UIToolPart | ui_message.UIDynamicToolPart,
) -> Any:
    if part.state == "output-error" and part.input is None:
        return part.raw_input
    return part.input


def _tool_result_output(
    part: ui_message.UIToolPart | ui_message.UIDynamicToolPart,
) -> Any:
    if part.state == "output-error":
        return _error_result(part.error_text, part.output)
    if part.state == "output-denied":
        reason = part.approval.reason if part.approval is not None else None
        return {
            "type": "error-text",
            "value": reason or "Tool call execution denied.",
        }
    return part.output


def _normalize_tool_result(output: Any) -> dict[str, Any] | None:
    """Normalize tool output to dict format for internal ToolResultPart."""
    if output is None:
        return None
    return output if isinstance(output, dict) else {"value": output}


def _error_result(error_text: str | None, output: Any) -> dict[str, Any] | None:
    normalized = _normalize_tool_result(output)
    if error_text:
        if normalized is None:
            return {"error": error_text}
        if isinstance(normalized, dict) and "error" not in normalized:
            return {"error": error_text, **normalized}
    return normalized


def _decode_wire_output(output: Any) -> Any:
    """Reconstruct the internal snapshot type from a wire tool output.

    Hacky special case: when the wire output looks like a ``UIMessage``
    (the wire shape we emit for sub-agent / ``MessageAggregator`` tools),
    decode it back to a ``MessageBundle``.  Other shapes pass through
    unchanged.  This avoids requiring callers to thread the tool
    registry into inbound parsing.
    """
    if not isinstance(output, dict):
        return output
    if output.get("role") != "assistant" or "parts" not in output:
        return output
    try:
        ui_msg = ui_message.UIMessage.model_validate(output)
    except Exception:
        return output
    inner = list(_parse([ui_msg]))
    return MessageBundle(messages=tuple(inner))


def _approval_hook_part(
    tp: ui_message.UIToolPart | ui_message.UIDynamicToolPart,
) -> messages_.HookPart[Any] | None:
    """Reconstruct approval hook state from a UI tool part when possible."""
    approval = tp.approval
    if approval is None:
        return None
    metadata = _approval_metadata(tp)

    if tp.state == "approval-requested":
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type="ToolApproval",
            status="pending",
            metadata=metadata,
        )

    if tp.state == "approval-responded" and approval.approved is not None:
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type="ToolApproval",
            status="resolved",
            metadata=metadata,
            resolution={
                "granted": approval.approved,
                "reason": approval.reason,
            },
        )

    if tp.state == "output-denied":
        return messages_.HookPart(
            hook_id=approval.id,
            hook_type="ToolApproval",
            status="resolved",
            metadata=metadata,
            resolution={
                "granted": False,
                "reason": approval.reason,
            },
        )

    return None


def _approval_metadata(
    tp: ui_message.UIToolPart | ui_message.UIDynamicToolPart,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if tp.approval is not None and tp.approval.is_automatic is not None:
        metadata["isAutomatic"] = tp.approval.is_automatic
    if tp.provider_executed is not None:
        metadata["providerExecuted"] = tp.provider_executed
    if tp.call_provider_metadata is not None:
        metadata["callProviderMetadata"] = tp.call_provider_metadata
    return metadata


# ============================================================================
# Approval extraction + bulk resolution
# ============================================================================


class ApprovalResponse(NamedTuple):
    """Approval response extracted from a responded UIToolPart."""

    hook_id: str
    granted: bool
    reason: str | None
    tool_call_id: str


def extract_approvals(
    ui_messages: list[ui_message.UIMessage],
) -> list[ApprovalResponse]:
    """Return every approval response found in *ui_messages*.

    Pure function — does not resolve hooks or trigger side effects.
    """
    approvals: list[ApprovalResponse] = []
    for ui_msg in ui_messages:
        for part in ui_msg.parts:
            if not isinstance(
                part, ui_message.UIToolPart | ui_message.UIDynamicToolPart
            ):
                continue
            if (
                part.state == "approval-responded"
                and part.approval is not None
                and part.approval.approved is not None
            ):
                approvals.append(
                    ApprovalResponse(
                        hook_id=part.approval.id,
                        granted=part.approval.approved,
                        reason=part.approval.reason,
                        tool_call_id=part.tool_call_id,
                    )
                )
    return approvals


def apply_approvals(approvals: list[ApprovalResponse]) -> None:
    """Pre-register each approval resolution with the hooks registry."""
    for approval in approvals:
        resolve_hook(
            approval.hook_id,
            {"granted": approval.granted, "reason": approval.reason},
        )


# ============================================================================
# UI message normalization (heal stale tool states)
# ============================================================================


def _normalize_ui_messages(
    ui_messages: list[ui_message.UIMessage],
) -> list[ui_message.UIMessage]:
    """Heal stale tool-part states from persisted assistant history."""
    normalized: list[ui_message.UIMessage] = []
    for message in ui_messages:
        new_parts = []
        changed = False
        for part in message.parts:
            part_type = getattr(part, "type", None)
            state = getattr(part, "state", None)
            if isinstance(part_type, str) and (
                part_type.startswith("tool-") or part_type == "dynamic-tool"
            ):
                output = getattr(part, "output", None)
                approval = getattr(part, "approval", None)
                approved = approval.approved if approval is not None else None
                error_text = getattr(part, "error_text", None)

                next_state = state
                if output is not None:
                    if state == "output-error" or error_text is not None:
                        next_state = "output-error"
                    elif state == "output-denied" or approved is False:
                        next_state = "output-denied"
                    else:
                        next_state = "output-available"
                elif state == "call":
                    next_state = "input-available"

                if next_state != state:
                    part = part.model_copy(update={"state": next_state})
                    changed = True

            new_parts.append(part)

        normalized.append(
            message.model_copy(update={"parts": new_parts})
            if changed
            else message
        )
    return normalized


# ============================================================================
# UI → internal message conversion
# ============================================================================


def to_messages(
    ui_messages: list[ui_message.UIMessage],
) -> tuple[list[messages_.Message], list[ApprovalResponse]]:
    """Parse a UI request into runtime messages + extracted approvals.

    Pure: normalizes stale tool states, extracts approval responses,
    parses UIMessages into an ``ai.messages.Message`` list (split at
    tool boundaries), drops the internal tombstones for approval
    responses, and patches the trailing tool message with
    ``is_hook_pending`` placeholders for tool calls whose approval was
    just responded to but never recorded a real tool result.

    Sub-agent tool outputs (UIMessage wire shape) are decoded back to
    ``MessageBundle`` so the parent agent's message history carries the
    rich snapshot.  Per-tool model-facing values are populated by
    :meth:`Agent.run` (which has the tool registry), not here.

    Returns ``(messages, approvals)``.  The caller can pre-register
    resolutions via :func:`apply_approvals` before calling
    :meth:`Agent.run` if the run should resume from a hook.
    """
    normalized = _normalize_ui_messages(ui_messages)
    approvals = extract_approvals(normalized)
    messages = [m for m in _parse(normalized) if not _is_approval_response(m)]
    _patch_pending_hook_aborts(messages, approvals)
    return messages, approvals


def _patch_pending_hook_aborts(
    messages: list[messages_.Message],
    approvals: list[ApprovalResponse],
) -> None:
    """Inject pending-hook placeholders for unresolved tool calls.

    This handles tool calls whose approval was responded to but whose tool
    result is still missing.

    This deals with the case where a prior run emitted multiple tool
    calls, some of which succeeded and some of which aborted on an
    approval hook.

    In that case, there will be an assistant message with multiple
    tool calls, a tool result with fewer results (some are missing),
    and then also some hook approvals.

    This synthesizes `ToolResultPart`s with `is_hook_pending=True` in
    order to be able to feed things back into the agent protocol.
    """
    if len(messages) < 2:
        return

    tool_msg = messages[-1]
    assistant_msg = messages[-2]
    if tool_msg.role != "tool" or assistant_msg.role != "assistant":
        return
    if not assistant_msg.tool_calls:
        return

    hooks = {a.tool_call_id: a for a in approvals}
    completed_ids = {r.tool_call_id for r in tool_msg.tool_results}

    new_parts: list[messages_.Part] = list(tool_msg.parts)
    for tc in assistant_msg.tool_calls:
        if tc.tool_call_id in completed_ids:
            continue
        if not (hook := hooks.get(tc.tool_call_id)):
            continue
        new_parts.append(
            messages_.ToolResultPart(
                id=f"{tool_msg.id}:result:{tc.tool_call_id}:hook-pending",
                tool_call_id=tc.tool_call_id,
                tool_name=tc.tool_name,
                result=f"Pending on hook '{hook.hook_id}'",
                is_error=True,
                is_hook_pending=True,
            )
        )

    if len(new_parts) != len(tool_msg.parts):
        messages[-1] = tool_msg.model_copy(update={"parts": new_parts})


def _is_approval_response(msg: messages_.Message) -> bool:
    """Return whether ``msg`` records a resolved tool-approval hook."""
    if msg.role != "internal" or len(msg.parts) != 1:
        return False
    part = msg.parts[0]
    return (
        isinstance(part, messages_.HookPart)
        and part.hook_type == "ToolApproval"
        and part.status == "resolved"
    )


def _parse(
    ui_messages: list[ui_message.UIMessage],
) -> list[messages_.Message]:
    def _build_result_part(
        *,
        tool_call_id: str,
        tool_name: str,
        output: Any,
        is_error: bool,
    ) -> messages_.ToolResultPart:
        if is_error:
            result: Any = output
        else:
            decoded = _decode_wire_output(output)
            result = (
                decoded
                if isinstance(decoded, MessageBundle)
                else (_normalize_tool_result(decoded))
            )
        return messages_.ToolResultPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
            is_error=is_error,
        )

    def _build_builtin_return_part(
        *,
        tool_call_id: str,
        tool_name: str,
        output: Any,
        is_error: bool,
        provider_metadata: dict[str, Any] | None,
    ) -> messages_.BuiltinToolReturnPart:
        return messages_.BuiltinToolReturnPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=output,
            is_error=is_error,
            provider_metadata=provider_metadata,
        )

    result: list[messages_.Message] = []

    for ui_msg in ui_messages:
        assistant_parts: list[messages_.Part] = []
        tool_result_parts: list[messages_.ToolResultPart] = []
        hook_parts: list[messages_.HookPart[Any]] = []

        for part in ui_msg.parts:
            match part:
                case ui_message.UITextPart(text=text) if text:
                    assistant_parts.append(
                        messages_.TextPart(
                            text=text,
                            provider_metadata=part.provider_metadata,
                        )
                    )

                case ui_message.UIReasoningPart(text=reasoning) if reasoning:
                    assistant_parts.append(
                        messages_.ReasoningPart(
                            text=reasoning,
                            provider_metadata=part.provider_metadata,
                        )
                    )

                case ui_message.UIToolInvocationPart() as inv:
                    tool_args = json.dumps(inv.args) if inv.args else "{}"
                    if inv.provider_executed:
                        assistant_parts.append(
                            messages_.BuiltinToolCallPart(
                                tool_call_id=inv.tool_invocation_id,
                                tool_name=inv.tool_name,
                                tool_args=tool_args,
                            )
                        )
                        if _is_tool_completed(inv.state):
                            assistant_parts.append(
                                _build_builtin_return_part(
                                    tool_call_id=inv.tool_invocation_id,
                                    tool_name=inv.tool_name,
                                    output=inv.result,
                                    is_error=_is_tool_error(inv.state),
                                    provider_metadata=None,
                                )
                            )
                    else:
                        assistant_parts.append(
                            messages_.ToolCallPart(
                                tool_call_id=inv.tool_invocation_id,
                                tool_name=inv.tool_name,
                                tool_args=tool_args,
                            )
                        )
                        if _is_tool_completed(inv.state):
                            tool_result_parts.append(
                                _build_result_part(
                                    tool_call_id=inv.tool_invocation_id,
                                    tool_name=inv.tool_name,
                                    output=inv.result,
                                    is_error=_is_tool_error(inv.state),
                                )
                            )

                case (
                    ui_message.UIToolPart() | ui_message.UIDynamicToolPart()
                ) as tp:
                    tool_input = _tool_input_for_args(tp)
                    tool_args = _normalize_tool_args(tool_input)

                    if tp.provider_executed:
                        assistant_parts.append(
                            messages_.BuiltinToolCallPart(
                                tool_call_id=tp.tool_call_id,
                                tool_name=tp.tool_name,
                                tool_args=tool_args,
                                provider_metadata=tp.call_provider_metadata,
                            )
                        )
                    else:
                        assistant_parts.append(
                            messages_.ToolCallPart(
                                tool_call_id=tp.tool_call_id,
                                tool_name=tp.tool_name,
                                tool_args=tool_args,
                                provider_metadata=tp.call_provider_metadata,
                            )
                        )
                    approval_hook = _approval_hook_part(tp)
                    if approval_hook is not None:
                        hook_parts.append(approval_hook)

                    if tp.provider_executed and _is_tool_completed(tp.state):
                        assistant_parts.append(
                            _build_builtin_return_part(
                                tool_call_id=tp.tool_call_id,
                                tool_name=tp.tool_name,
                                output=_tool_result_output(tp),
                                is_error=_is_tool_error(tp.state),
                                provider_metadata=(
                                    tp.result_provider_metadata
                                    or tp.call_provider_metadata
                                ),
                            )
                        )
                    elif tp.state in _TOOL_RESULT_STATES:
                        tool_result_parts.append(
                            _build_result_part(
                                tool_call_id=tp.tool_call_id,
                                tool_name=tp.tool_name,
                                output=_tool_result_output(tp),
                                is_error=_is_tool_error(tp.state),
                            )
                        )
                        if tp.result_provider_metadata is not None:
                            tool_result_parts[-1] = tool_result_parts[
                                -1
                            ].model_copy(
                                update={
                                    "provider_metadata": (
                                        tp.result_provider_metadata
                                    )
                                }
                            )

                case ui_message.UIFilePart() as fp:
                    assistant_parts.append(
                        messages_.FilePart(
                            data=fp.url,
                            media_type=fp.media_type,
                            filename=fp.filename,
                            provider_metadata=fp.provider_metadata,
                        )
                    )

                case (
                    ui_message.UIStepStartPart()
                    | ui_message.UISourceUrlPart()
                    | ui_message.UISourceDocumentPart()
                    | ui_message.UIReasoningFilePart()
                    | ui_message.UICustomPart()
                    | ui_message.UIDataPart()
                ):
                    pass

        if ui_msg.role in ("user", "system") and not assistant_parts:
            raise ValueError(
                f"Message {ui_msg.id!r} has role {ui_msg.role!r} "
                "but no content. "
                "User and system messages require non-empty content."
            )

        # The UI sends one assistant message per conversation turn, but a
        # single turn may span multiple loop iterations (e.g. [text,
        # tool_call, tool_result, text, tool_call, tool_result, text]).
        # LLM APIs expect one message per iteration, so split into
        # assistant + tool message pairs at tool-result boundaries.
        if ui_msg.role == "assistant":
            result.extend(
                _split_assistant_parts(
                    assistant_parts,
                    tool_result_parts,
                    turn_id=ui_msg.id,
                )
            )
            for index, hp in enumerate(hook_parts):
                msg_id = _AdapterIdPolicy.message_id(
                    ui_msg.id, "internal", index
                )
                result.append(
                    messages_.Message(
                        id=msg_id,
                        turn_id=ui_msg.id,
                        role="internal",
                        parts=_AdapterIdPolicy.with_stable_part_ids(
                            msg_id, [hp]
                        ),
                    )
                )
        else:
            result.append(
                messages_.Message(
                    id=ui_msg.id,
                    role=ui_msg.role,
                    parts=_AdapterIdPolicy.with_stable_part_ids(
                        ui_msg.id, assistant_parts
                    ),
                )
            )

    return result


def _split_assistant_parts(
    parts: list[messages_.Part],
    tool_results: list[messages_.ToolResultPart],
    turn_id: str,
) -> list[messages_.Message]:
    """Split assistant parts into assistant + tool message pairs."""
    results_by_id = {tr.tool_call_id: tr for tr in tool_results}

    pending_results: list[messages_.ToolResultPart] = []
    for part in parts:
        if (
            isinstance(part, messages_.ToolCallPart)
            and part.tool_call_id in results_by_id
        ):
            pending_results.append(results_by_id[part.tool_call_id])

    if not pending_results:
        if parts:
            msg_id = _AdapterIdPolicy.message_id(turn_id, "assistant", 0)
            return [
                messages_.Message(
                    role="assistant",
                    parts=_AdapterIdPolicy.with_stable_part_ids(msg_id, parts),
                    id=msg_id,
                    turn_id=turn_id,
                )
            ]
        return []

    messages: list[messages_.Message] = []
    current: list[messages_.Part] = []
    current_results: list[messages_.ToolResultPart] = []
    seen_tool_call = False
    assistant_index = 0
    tool_index = 0

    def _append_assistant(parts_: list[messages_.Part]) -> None:
        nonlocal assistant_index
        msg_id = _AdapterIdPolicy.message_id(
            turn_id, "assistant", assistant_index
        )
        messages.append(
            messages_.Message(
                role="assistant",
                parts=_AdapterIdPolicy.with_stable_part_ids(msg_id, parts_),
                id=msg_id,
                turn_id=turn_id,
            )
        )
        assistant_index += 1

    def _append_tool(parts_: list[messages_.ToolResultPart]) -> None:
        nonlocal tool_index
        msg_id = _AdapterIdPolicy.message_id(turn_id, "tool", tool_index)
        messages.append(
            messages_.Message(
                role="tool",
                parts=_AdapterIdPolicy.with_stable_part_ids(
                    msg_id, list(parts_)
                ),
                id=msg_id,
                turn_id=turn_id,
            )
        )
        tool_index += 1

    for part in parts:
        if (
            seen_tool_call
            and current_results
            and not isinstance(part, messages_.ToolCallPart)
        ):
            _append_assistant(current)
            _append_tool(current_results)
            current = []
            current_results = []
            seen_tool_call = False

        current.append(part)

        if isinstance(part, messages_.ToolCallPart):
            seen_tool_call = True
            if part.tool_call_id in results_by_id:
                current_results.append(results_by_id[part.tool_call_id])

    if current:
        _append_assistant(current)
    if current_results:
        _append_tool(current_results)

    return messages
