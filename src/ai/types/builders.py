"""Composable message construction helpers.

Convenience functions for building Message objects without manually
constructing Part lists. Each ``*_message`` function returns a single
``Message``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import events as events_

from .messages import (
    ContentOutput,
    ContentPart,
    FilePart,
    HookPart,
    Message,
    Part,
    ReasoningPart,
    ResultKind,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

_PART_TYPES = (
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    HookPart,
    FilePart,
)

# A value that can appear as message content: bare strings become TextPart.
PartLike = str | Part


def _coerce_parts(args: tuple[PartLike, ...]) -> list[Part]:
    parts: list[Part] = []
    for arg in args:
        if isinstance(arg, str):
            parts.append(TextPart(text=arg))
        elif isinstance(arg, _PART_TYPES):
            parts.append(arg)
        else:
            raise TypeError(f"Expected str or Part, got {type(arg).__name__}")
    return parts


def system_message(*content: PartLike) -> Message:
    """Create a system message.

    >>> ai.system_message("You are a helpful robot.")
    """
    return Message(role="system", parts=_coerce_parts(content))


def user_message(*content: PartLike) -> Message:
    """Create a user message from strings and/or Part objects.

    >>> ai.user_message("Describe this image:", ai.file_part(url))
    """
    return Message(role="user", parts=_coerce_parts(content))


def assistant_message(*content: PartLike) -> Message:
    """Create an assistant message from strings and/or Part objects.

    >>> ai.assistant_message(ai.thinking("hmm"), "Hello!")
    """
    return Message(role="assistant", parts=_coerce_parts(content))


def file_part(
    data: str | bytes,
    *,
    media_type: str | None = None,
    filename: str | None = None,
) -> FilePart:
    """Create a :class:`FilePart` from a URL string or raw bytes.

    Dispatches to :meth:`FilePart.from_url` (for ``str``) or
    :meth:`FilePart.from_bytes` (for ``bytes``), with automatic
    media-type detection.
    """
    if isinstance(data, str):
        return FilePart.from_url(data, media_type=media_type)
    return FilePart.from_bytes(data, media_type=media_type, filename=filename)


def text_part(
    text: str,
    *,
    provider_metadata: dict[str, Any] | None = None,
) -> TextPart:
    """Create a :class:`TextPart`.

    Bare strings passed to the ``*_message`` builders are coerced into
    text parts automatically; reach for this when you need to attach
    ``provider_metadata`` or build a part list directly.
    """
    return TextPart(text=text, provider_metadata=provider_metadata)


def content_output(*content: str | TextPart | FilePart) -> ContentOutput:
    """Create a multipart :class:`ContentOutput` tool result.

    Bare strings become :class:`TextPart` objects, mirroring the
    ``*_message`` builders, so a tool can return mixed text and files
    without constructing the part list by hand.

    >>> ai.content_output("Here is the chart:", ai.file_part(png_bytes))
    """
    parts: list[ContentPart] = []
    for item in content:
        parts.append(TextPart(text=item) if isinstance(item, str) else item)
    return ContentOutput(value=parts)


def thinking(
    text: str,
    *,
    provider_metadata: dict[str, Any] | None = None,
) -> ReasoningPart:
    """Create a :class:`ReasoningPart`.

    Useful for replaying conversation history that includes model reasoning.
    """
    return ReasoningPart(text=text, provider_metadata=provider_metadata)


def _tool_results_from_messages(
    messages: list[Message],
) -> list[ToolResultPart]:
    parts: list[ToolResultPart] = []
    for message in messages:
        if message.role != "tool":
            raise TypeError(f"Expected tool message, got role={message.role!r}")
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                raise TypeError(
                    "tool_message() only accepts tool messages containing "
                    "ToolResultPart parts"
                )
            parts.append(part)
    return parts


def tool_message(
    *items: Message | ToolResultPart | events_.ToolCallResult | list[Message],
    tool_call_id: str | None = None,
    result: Any = None,
    tool_name: str = "",
    is_error: bool = False,
) -> Message:
    """Create or merge a tool-result message.

    >>> part = ai.tool_result_part("tc-1", result=72, tool_name="weather")
    >>> ai.tool_message(part)
    >>> ai.tool_message(tool_call_id="tc-1", result=72, tool_name="weather")
    """
    if tool_call_id is None and (result is not None or tool_name or is_error):
        raise TypeError(
            "tool_message() keyword tool-result fields require tool_call_id"
        )

    if tool_call_id is not None:
        if items:
            raise TypeError(
                "tool_message() cannot mix keyword tool-result fields with "
                "positional messages or ToolResultPart values"
            )
        return Message(
            role="tool",
            parts=[
                tool_result_part(
                    tool_call_id,
                    result=result,
                    tool_name=tool_name,
                    is_error=is_error,
                )
            ],
        )

    if not items:
        raise TypeError(
            "tool_message() requires at least one tool message or result"
        )

    flattened_messages: list[Message] = []
    result_parts: list[ToolResultPart] = []
    saw_message = False
    saw_result_part = False

    for item in items:
        if isinstance(item, list):
            saw_message = True
            flattened_messages.extend(item)
        elif isinstance(item, Message):
            saw_message = True
            flattened_messages.append(item)
        elif isinstance(item, ToolResultPart):
            saw_result_part = True
            result_parts.append(item)
        elif hasattr(item, "message") and isinstance(item.message, Message):
            # ToolCallResult — can't isinstance-check due to circular import
            saw_message = True
            flattened_messages.append(item.message)
        else:
            raise TypeError(
                "tool_message() only accepts tool messages, lists of tool "
                "messages, ToolResultPart, or ToolCallResult values"
            )

    if saw_message and saw_result_part:
        raise TypeError(
            "tool_message() cannot mix tool messages with ToolResultPart values"
        )

    if saw_message:
        merged_parts: list[Part] = []
        merged_parts.extend(_tool_results_from_messages(flattened_messages))
        return Message(role="tool", parts=merged_parts)

    tool_parts: list[Part] = []
    tool_parts.extend(result_parts)
    return Message(role="tool", parts=tool_parts)


def tool_result_part(
    tool_call_id: str,
    *,
    result: Any = None,
    tool_name: str = "",
    is_error: bool = False,
) -> ToolResultPart:
    """Create a :class:`ToolResultPart`.

    ``result`` is stored as-is; ``result_kind`` is derived: ``"error"`` when
    ``is_error`` is set, ``"special"`` for a :class:`ContentOutput` or
    :class:`MessageBundle`, else ``"json"`` (a ``str`` is sent raw to the
    model, anything else is JSON-encoded at the provider boundary).

    >>> ai.tool_result_part("tc-1", result={"temp": 72}, tool_name="weather")
    """
    result_kind: ResultKind = (
        "error" if is_error else ToolResultPart.kind_for(result)
    )
    return ToolResultPart(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=result,
        result_kind=result_kind,
    )
