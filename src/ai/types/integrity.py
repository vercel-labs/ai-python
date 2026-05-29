import json
import logging
from typing import Literal

from . import builders
from . import messages as messages_

logger = logging.getLogger(__name__)

Mode = Literal["strict", "auto"]

IssueKind = Literal[
    "duplicate-tool-call",
    "duplicate-tool-result",
    "internal-part",
    "invalid-tool-args",
    "orphaned-tool-call",
    "orphaned-tool-result",
    "internal-message",
]


class IntegrityError(ValueError):
    def __init__(self, issues: list[IssueKind]) -> None:
        self.issues = issues
        super().__init__(
            f"Message history has {len(issues)} issue(s): " + ", ".join(issues)
        )


# used for stripping internal parts
_LLM_PART_TYPES = (
    messages_.TextPart,
    messages_.ToolCallPart,
    messages_.ToolResultPart,
    messages_.BuiltinToolCallPart,
    messages_.BuiltinToolReturnPart,
    messages_.ReasoningPart,
    messages_.FilePart,
)


def _clean_messages(
    messages: list[messages_.Message], mode: Mode
) -> tuple[list[messages_.Message], list[IssueKind]]:
    """Strip internal messages, fix broken tool args."""
    issues: list[IssueKind] = []
    result: list[messages_.Message] = []

    for msg in messages:
        # 1. drop internal messages emitted by hooks
        if msg.role == "internal":
            issues.append("internal-message")
            if mode == "strict":
                result.append(msg)
            continue

        parts: list[messages_.Part] = list(msg.parts)
        changed = False

        # 2. strip everything that isn't an LLM part
        kept: list[messages_.Part] = [
            p for p in parts if isinstance(p, _LLM_PART_TYPES)
        ]
        if len(kept) < len(parts):
            issues.append("internal-part")
            if mode == "auto":
                parts = kept
                changed = True

        # 3. ensure tool args are json-decodable
        new_parts: list[messages_.Part] = []
        for part in parts:
            if isinstance(part, messages_.ToolCallPart):
                try:
                    json.loads(part.tool_args)
                except (json.JSONDecodeError, TypeError):
                    if mode == "auto":
                        part = part.model_copy(update={"tool_args": "{}"})
                    issues.append("invalid-tool-args")
                    changed = True
            new_parts.append(part)

        if changed and mode == "auto":
            parts = new_parts

        # 4. drop empty messages
        if mode == "auto" and not parts:
            continue

        if changed and mode == "auto":
            # messages are immutable so we have to do this
            result.append(msg.model_copy(update={"parts": parts}))
        else:
            result.append(msg)

    return result, issues


def _validate_tool_ids(messages: list[messages_.Message]) -> list[IssueKind]:
    """Check for fatal issues: duplicate tool ids, orphaned tool results."""
    issues: list[IssueKind] = []
    seen_call_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    pending_call_ids: set[str] = set()

    duplicate_call = False
    duplicate_result = False
    orphaned_result = False

    for msg in messages:
        if msg.role in ("user", "assistant") and pending_call_ids:
            # result should have been in a tool message before this
            # if it wasn't then it's a stray call, will be auto-fixed later
            pending_call_ids.clear()

        if msg.role == "assistant":
            # check if tool call is duplicate
            # if not, mark it and append it to pending
            for part in msg.parts:
                if not isinstance(part, messages_.ToolCallPart):
                    continue
                if part.tool_call_id in seen_call_ids:
                    duplicate_call = True
                else:
                    seen_call_ids.add(part.tool_call_id)
                pending_call_ids.add(part.tool_call_id)

        elif msg.role == "tool":
            # check that this tool result is not duplicate and that
            # there's a pending call from previous assistant message
            for part in msg.parts:
                if not isinstance(part, messages_.ToolResultPart):
                    continue
                if part.tool_call_id in seen_result_ids:
                    duplicate_result = True
                else:
                    seen_result_ids.add(part.tool_call_id)
                if part.tool_call_id not in pending_call_ids:
                    orphaned_result = True
                    continue
                pending_call_ids.remove(part.tool_call_id)

    if duplicate_call:
        issues.append("duplicate-tool-call")
    if duplicate_result:
        issues.append("duplicate-tool-result")
    if orphaned_result:
        issues.append("orphaned-tool-result")

    return issues


def _fix_missing_results(
    messages: list[messages_.Message], mode: Mode
) -> tuple[list[messages_.Message], list[IssueKind]]:
    """Insert fake error results for stray tool calls."""
    issues: list[IssueKind] = []
    result: list[messages_.Message] = []

    # 1. collect all result ids
    answered: set[str] = set()
    for msg in messages:
        if msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, messages_.ToolResultPart):
                    answered.add(part.tool_call_id)

    # pending tool calls from the current assistant turn
    pending: dict[str, messages_.ToolCallPart] = {}

    def _flush_pending() -> None:
        if not pending:
            return
        issues.append("orphaned-tool-call")
        if mode == "auto":
            synthetic = builders.tool_message(
                *(
                    messages_.ToolResultPart(
                        tool_call_id=tc.tool_call_id,
                        tool_name=tc.tool_name,
                        result="Tool result not available",
                        result_kind="error",
                    )
                    for tc in pending.values()
                )
            )
            result.append(synthetic)

    for msg in messages:
        # if we're seeing a user / assistant message, then
        # all pending tool calls are strays, because their results
        # should have followed immediately after in a tool message
        if msg.role in ("user", "assistant") and pending:
            _flush_pending()
            pending.clear()

        # 2. track calls
        if msg.role == "assistant":
            for part in msg.parts:
                if (
                    isinstance(part, messages_.ToolCallPart)
                    and part.tool_call_id not in answered
                ):
                    pending[part.tool_call_id] = part
            result.append(msg)
        # 3. match results with calls
        elif msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, messages_.ToolResultPart):
                    pending.pop(part.tool_call_id, None)
            result.append(msg)
        else:
            result.append(msg)

    _flush_pending()

    return result, issues


def prepare_messages(
    messages: list[messages_.Message],
    *,
    mode: Mode = "auto",
) -> list[messages_.Message]:
    """Fix and validate message list.

    ``"auto"`` (default) -- silently fixes recoverable issues (signal
    messages, internal parts, invalid tool args, missing tool results).
    ``"strict"`` -- collects every recoverable issue and raises
    :class:`IntegrityError`.

    Duplicate tool-call IDs, duplicate tool-result IDs, and orphaned
    tool results always raise :class:`IntegrityError` regardless of mode.

    Always returns a **new** list; never mutates the input.
    """
    issues: list[IssueKind] = []

    result, phase1_issues = _clean_messages(list(messages), mode)
    issues.extend(phase1_issues)

    # never auto-fixed
    fatal_issues = _validate_tool_ids(result)
    issues.extend(fatal_issues)

    if not fatal_issues:
        result, phase3_issues = _fix_missing_results(result, mode)
        issues.extend(phase3_issues)

    if fatal_issues or (mode == "strict" and issues):
        raise IntegrityError(issues)

    if issues:
        logger.warning(
            "Auto-fixed %d message issue(s): %s",
            len(issues),
            ", ".join(issues),
        )

    return result
