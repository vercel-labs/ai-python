"""Inbound adapter from AI SDK v6 UIMessages to internal messages.

The primary entry point is :func:`to_messages`, which bundles normalization,
approval extraction, parsing, and pre-registration of approval resolutions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ....types import messages as messages_
from ...agent import MessageBundle
from . import approvals, id_utils
from . import ui_messages as ui_messages_
from .approvals import ApprovalResponse, extract_approvals
from .tool_utils import normalize_tool_args

logger = logging.getLogger(__name__)


_TOOL_RESULT_STATES: frozenset[str] = frozenset({"output-available"})
_TOOL_ERROR_STATES: frozenset[str] = frozenset(
    {"output-error", "output-denied"}
)


def _is_tool_completed(state: ui_messages_.UIToolInvocationState) -> bool:
    return state in _TOOL_RESULT_STATES or state in _TOOL_ERROR_STATES


def _is_tool_error(state: ui_messages_.UIToolInvocationState) -> bool:
    return state in _TOOL_ERROR_STATES


def _tool_input_for_args(
    part: ui_messages_.UIToolPart | ui_messages_.UIDynamicToolPart,
) -> Any:
    if part.state == "output-error" and part.input is None:
        return part.raw_input
    return part.input


def _tool_result_output(
    part: ui_messages_.UIToolPart | ui_messages_.UIDynamicToolPart,
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
        ui_msg = ui_messages_.UIMessage.model_validate(output)
    except Exception:
        return output
    inner = list(_parse([ui_msg]))
    return MessageBundle(messages=tuple(inner))


def _normalize_ui_messages(
    ui_messages: list[ui_messages_.UIMessage],
) -> list[ui_messages_.UIMessage]:
    """Heal stale tool-part states from persisted assistant history."""
    normalized: list[ui_messages_.UIMessage] = []
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
    ui_messages: list[ui_messages_.UIMessage],
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
    approval_responses = extract_approvals(normalized)
    messages = [
        m
        for m in _parse(normalized)
        if not approvals.is_resolved_approval_message(m)
    ]
    _patch_pending_hook_aborts(messages, approval_responses)
    return messages, approval_responses


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
                tool_call_id=tc.tool_call_id,
                tool_name=tc.tool_name,
                result=f"Pending on hook '{hook.hook_id}'",
                is_error=True,
                is_hook_pending=True,
            )
        )

    if len(new_parts) != len(tool_msg.parts):
        messages[-1] = tool_msg.model_copy(update={"parts": new_parts})


def _parse(
    ui_messages: list[ui_messages_.UIMessage],
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
        source_messages = id_utils.source_messages_from(ui_msg.metadata)
        assistant_parts: list[messages_.Part] = []
        tool_result_parts: list[messages_.ToolResultPart] = []
        hook_parts: list[messages_.HookPart[Any]] = []

        for part in ui_msg.parts:
            match part:
                case ui_messages_.UITextPart(text=text) if text:
                    assistant_parts.append(
                        messages_.TextPart(
                            text=text,
                            provider_metadata=part.provider_metadata,
                        )
                    )

                case ui_messages_.UIReasoningPart(text=reasoning) if reasoning:
                    assistant_parts.append(
                        messages_.ReasoningPart(
                            text=reasoning,
                            provider_metadata=part.provider_metadata,
                        )
                    )

                case ui_messages_.UIToolInvocationPart() as inv:
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
                    (
                        ui_messages_.UIToolPart()
                        | ui_messages_.UIDynamicToolPart()
                    ) as tp
                ):
                    tool_input = _tool_input_for_args(tp)
                    tool_args = normalize_tool_args(tool_input)

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
                    approval_hook = approvals.hook_part_from_tool_part(tp)
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

                case ui_messages_.UIFilePart() as fp:
                    assistant_parts.append(
                        messages_.FilePart(
                            data=fp.url,
                            media_type=fp.media_type,
                            filename=fp.filename,
                            provider_metadata=fp.provider_metadata,
                        )
                    )

                case (
                    ui_messages_.UIStepStartPart()
                    | ui_messages_.UISourceUrlPart()
                    | ui_messages_.UISourceDocumentPart()
                    | ui_messages_.UIReasoningFilePart()
                    | ui_messages_.UICustomPart()
                    | ui_messages_.UIDataPart()
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
            parsed = _split_assistant_parts(
                assistant_parts,
                tool_result_parts,
                turn_id=ui_msg.id,
            )
            for hp in hook_parts:
                parsed.append(
                    messages_.Message(
                        turn_id=ui_msg.id,
                        role="internal",
                        parts=[hp],
                    )
                )
            result.extend(id_utils.restore_source_ids(parsed, source_messages))
        else:
            result.extend(
                id_utils.restore_source_ids(
                    [
                        messages_.Message(
                            id=ui_msg.id,
                            role=ui_msg.role,
                            parts=assistant_parts,
                        )
                    ],
                    source_messages,
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
            return [
                messages_.Message(
                    role="assistant",
                    parts=parts,
                    turn_id=turn_id,
                )
            ]
        return []

    messages: list[messages_.Message] = []
    current: list[messages_.Part] = []
    current_results: list[messages_.ToolResultPart] = []
    seen_tool_call = False

    def _append_assistant(parts_: list[messages_.Part]) -> None:
        messages.append(
            messages_.Message(
                role="assistant",
                parts=parts_,
                turn_id=turn_id,
            )
        )

    def _append_tool(parts_: list[messages_.ToolResultPart]) -> None:
        messages.append(
            messages_.Message(
                role="tool",
                parts=list(parts_),
                turn_id=turn_id,
            )
        )

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
