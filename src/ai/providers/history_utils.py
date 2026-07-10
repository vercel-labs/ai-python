"""Message-history utilities for provider implementations.

The core ``stream()`` / ``generate()`` functions pass message history to
providers untouched; deciding what reaches the LLM is the provider's
job.  Providers call :func:`repair` (or the individual fix functions)
before converting messages to their wire format.

Applications that want to fail fast instead of silently repairing can
call :func:`validate` (or :func:`inspect`) on a history themselves.
"""

import json
import logging
from typing import Literal

import pydantic

from ..types import builders
from ..types import messages as messages_

logger = logging.getLogger(__name__)

IssueKind = Literal[
    "internal-message",
    "internal-part",
    "invalid-tool-args",
    "orphaned-tool-call",
    "orphaned-tool-result",
    "duplicate-tool-call",
    "duplicate-tool-result",
]


class Issue(pydantic.BaseModel):
    kind: IssueKind
    message_id: str

    def __str__(self) -> str:
        return f"{self.kind} ({self.message_id})"


class IntegrityError(ValueError):
    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__(
            f"Message history has {len(issues)} issue(s): "
            + ", ".join(str(issue) for issue in issues)
        )


# parts an LLM can consume; everything else is app-internal
_LLM_PART_TYPES = (
    messages_.TextPart,
    messages_.ToolCallPart,
    messages_.ToolResultPart,
    messages_.BuiltinToolCallPart,
    messages_.BuiltinToolReturnPart,
    messages_.ReasoningPart,
    messages_.FilePart,
)


def drop_internal(
    messages: list[messages_.Message],
) -> tuple[list[messages_.Message], list[Issue]]:
    """Drop what the LLM should never see.

    Removes internal-role messages, strips non-LLM parts (hooks), and
    drops messages left with no parts.  Returns a new list.
    """
    issues: list[Issue] = []
    result: list[messages_.Message] = []

    for msg in messages:
        if msg.role == "internal":
            issues.append(Issue(kind="internal-message", message_id=msg.id))
            continue
        kept = [p for p in msg.parts if isinstance(p, _LLM_PART_TYPES)]
        stripped = len(kept) < len(msg.parts)
        if stripped:
            issues.append(Issue(kind="internal-part", message_id=msg.id))
        if not kept:
            continue
        if stripped:
            result.append(msg.model_copy(update={"parts": kept}))
        else:
            result.append(msg)

    return result, issues


def fix_tool_args(
    messages: list[messages_.Message],
) -> tuple[list[messages_.Message], list[Issue]]:
    """Replace tool-call args that aren't valid JSON with ``"{}"``.

    Returns a new list.
    """
    issues: list[Issue] = []
    result: list[messages_.Message] = []

    for msg in messages:
        parts: list[messages_.Part] = []
        changed = False
        for part in msg.parts:
            if isinstance(part, messages_.ToolCallPart):
                try:
                    json.loads(part.tool_args)
                except (json.JSONDecodeError, TypeError):
                    issues.append(
                        Issue(kind="invalid-tool-args", message_id=msg.id)
                    )
                    part = part.model_copy(update={"tool_args": "{}"})
                    changed = True
            parts.append(part)
        if changed:
            result.append(msg.model_copy(update={"parts": parts}))
        else:
            result.append(msg)

    return result, issues


def check_tool_ids(messages: list[messages_.Message]) -> list[Issue]:
    """Detect duplicate tool ids and orphaned tool results.

    These are never fixed: dropping or renaming parts of an assistant
    turn silently changes what the conversation means.
    """
    issues: list[Issue] = []
    seen_call_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    pending_call_ids: set[str] = set()

    for msg in messages:
        if msg.role in ("user", "assistant") and pending_call_ids:
            # results for these calls should have arrived in a tool
            # message before this; leftovers are orphaned *calls*,
            # which are fixable and not our concern here
            pending_call_ids.clear()

        if msg.role == "assistant":
            for part in msg.parts:
                if not isinstance(part, messages_.ToolCallPart):
                    continue
                if part.tool_call_id in seen_call_ids:
                    issues.append(
                        Issue(kind="duplicate-tool-call", message_id=msg.id)
                    )
                else:
                    seen_call_ids.add(part.tool_call_id)
                pending_call_ids.add(part.tool_call_id)

        elif msg.role == "tool":
            for part in msg.parts:
                if not isinstance(part, messages_.ToolResultPart):
                    continue
                if part.tool_call_id in seen_result_ids:
                    issues.append(
                        Issue(kind="duplicate-tool-result", message_id=msg.id)
                    )
                else:
                    seen_result_ids.add(part.tool_call_id)
                if part.tool_call_id in pending_call_ids:
                    pending_call_ids.remove(part.tool_call_id)
                else:
                    issues.append(
                        Issue(kind="orphaned-tool-result", message_id=msg.id)
                    )

    return issues


def close_orphaned_tool_calls(
    messages: list[messages_.Message],
) -> tuple[list[messages_.Message], list[Issue]]:
    """Insert synthetic error results for tool calls that have none.

    A synthetic tool message is placed where the real one should have
    been: before the message that interrupted the turn, or at the end
    of the history.  Returns a new list.
    """
    issues: list[Issue] = []
    result: list[messages_.Message] = []

    answered: set[str] = {
        part.tool_call_id
        for msg in messages
        if msg.role == "tool"
        for part in msg.parts
        if isinstance(part, messages_.ToolResultPart)
    }

    # unanswered tool calls from the current assistant turn,
    # tool_call_id -> (call part, id of the message it came from)
    pending: dict[str, tuple[messages_.ToolCallPart, str]] = {}

    def _flush_pending() -> None:
        if not pending:
            return
        issues.extend(
            Issue(kind="orphaned-tool-call", message_id=msg_id)
            for _, msg_id in pending.values()
        )
        result.append(
            builders.tool_message(
                *(
                    messages_.ToolResultPart(
                        tool_call_id=tc.tool_call_id,
                        tool_name=tc.tool_name,
                        result="Tool result not available",
                        result_kind="error",
                    )
                    for tc, _ in pending.values()
                )
            )
        )
        pending.clear()

    for msg in messages:
        # a user or assistant message means the previous turn's tool
        # calls can no longer be answered: their results had to come
        # in a tool message right before this one
        if msg.role in ("user", "assistant"):
            _flush_pending()

        if msg.role == "assistant":
            for part in msg.parts:
                if (
                    isinstance(part, messages_.ToolCallPart)
                    and part.tool_call_id not in answered
                ):
                    pending[part.tool_call_id] = (part, msg.id)
        elif msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, messages_.ToolResultPart):
                    pending.pop(part.tool_call_id, None)
        result.append(msg)

    _flush_pending()

    return result, issues


def inspect(messages: list[messages_.Message]) -> list[Issue]:
    """Report every integrity issue in the history without changing it."""
    result, issues = drop_internal(messages)
    result, args_issues = fix_tool_args(result)
    issues.extend(args_issues)
    id_issues = check_tool_ids(result)
    issues.extend(id_issues)
    if not id_issues:
        # orphan detection is meaningless while tool ids are broken
        issues.extend(close_orphaned_tool_calls(result)[1])
    return issues


def validate(messages: list[messages_.Message]) -> None:
    """Raise :class:`IntegrityError` if the history has any issue."""
    issues = inspect(messages)
    if issues:
        raise IntegrityError(issues)


def repair(messages: list[messages_.Message]) -> list[messages_.Message]:
    """Fix what can be fixed, raise on what can't.

    Runs :func:`drop_internal`, :func:`fix_tool_args`, and
    :func:`close_orphaned_tool_calls`, logging a warning for every fix.
    Raises :class:`IntegrityError` on duplicate tool-call ids, duplicate
    tool-result ids, and orphaned tool results — there is no safe
    automatic fix for those.

    A convenience wrapper: call the individual fix functions instead
    when you need the issues themselves.

    Always returns a new list; never mutates the input.
    """
    result, issues = drop_internal(messages)
    result, args_issues = fix_tool_args(result)
    issues.extend(args_issues)

    fatal = check_tool_ids(result)
    if fatal:
        raise IntegrityError(fatal)

    result, orphan_issues = close_orphaned_tool_calls(result)
    issues.extend(orphan_issues)

    if issues:
        logger.warning(
            "Repaired %d message issue(s): %s",
            len(issues),
            ", ".join(str(issue) for issue in issues),
        )

    return result
