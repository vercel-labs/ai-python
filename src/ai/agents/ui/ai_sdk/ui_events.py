from __future__ import annotations

import dataclasses
from typing import Any, Literal

# necessary headers for the streaming integration to work
UI_MESSAGE_STREAM_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "x-vercel-ai-ui-message-stream": "v1",
    "x-accel-buffering": "no",
}


# different kinds of stream events expected by the frontend

FinishReason = Literal[
    "stop", "length", "content-filter", "tool-calls", "error", "other"
]


@dataclasses.dataclass
class UIStartEvent:
    """Indicates the beginning of a new message with metadata."""

    type: Literal["start"] = dataclasses.field(default="start", init=False)
    message_id: str | None = None
    message_metadata: Any | None = None


@dataclasses.dataclass
class UITextStartEvent:
    """Indicates the beginning of a text block."""

    id: str
    type: Literal["text-start"] = dataclasses.field(
        default="text-start", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UITextDeltaEvent:
    """Contains incremental text content for the text block."""

    id: str
    delta: str
    type: Literal["text-delta"] = dataclasses.field(
        default="text-delta", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UITextEndEvent:
    """Indicates the completion of a text block."""

    id: str
    type: Literal["text-end"] = dataclasses.field(
        default="text-end", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UIReasoningStartEvent:
    """Indicates the beginning of a reasoning block."""

    id: str
    type: Literal["reasoning-start"] = dataclasses.field(
        default="reasoning-start", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UIReasoningDeltaEvent:
    """Contains incremental reasoning content for the reasoning block."""

    id: str
    delta: str
    type: Literal["reasoning-delta"] = dataclasses.field(
        default="reasoning-delta", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UIReasoningEndEvent:
    """Indicates the completion of a reasoning block."""

    id: str
    type: Literal["reasoning-end"] = dataclasses.field(
        default="reasoning-end", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UICustomEvent:
    """Provider-specific content that does not fit standard UI events."""

    kind: str
    type: Literal["custom"] = dataclasses.field(default="custom", init=False)
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UISourceUrlEvent:
    """References to external URLs."""

    source_id: str
    url: str
    type: Literal["source-url"] = dataclasses.field(
        default="source-url", init=False
    )
    title: str | None = None
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UISourceDocumentEvent:
    """References to documents or files."""

    source_id: str
    media_type: str
    title: str
    type: Literal["source-document"] = dataclasses.field(
        default="source-document", init=False
    )
    filename: str | None = None
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UIFileEvent:
    """References to files with their media type."""

    url: str
    media_type: str
    type: Literal["file"] = dataclasses.field(default="file", init=False)
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UIReasoningFileEvent:
    """A file generated as part of model reasoning."""

    url: str
    media_type: str
    type: Literal["reasoning-file"] = dataclasses.field(
        default="reasoning-file", init=False
    )
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UIDataEvent:
    """Custom data event for arbitrary structured data.

    Data events support type-specific handling.

    The wire type is ``data-{data_type}`` (e.g. ``data-custom``), exposed
    via the ``type`` property so that ``UIDataEvent`` is uniform with every
    other ``UIMessageStreamEvent`` variant.
    """

    data_type: str
    data: Any
    id: str | None = None
    transient: bool | None = None

    @property
    def type(self) -> str:
        """Wire type for the AI SDK SSE protocol."""
        return f"data-{self.data_type}"


@dataclasses.dataclass
class UIToolInputStartEvent:
    """Indicates the beginning of tool input streaming."""

    tool_call_id: str
    tool_name: str
    type: Literal["tool-input-start"] = dataclasses.field(
        default="tool-input-start", init=False
    )
    provider_executed: bool | None = None
    provider_metadata: dict[str, Any] | None = None
    tool_metadata: dict[str, Any] | None = None
    dynamic: bool | None = None
    title: str | None = None


@dataclasses.dataclass
class UIToolInputDeltaEvent:
    """Incremental chunks of tool input as it's being generated."""

    tool_call_id: str
    input_text_delta: str
    type: Literal["tool-input-delta"] = dataclasses.field(
        default="tool-input-delta", init=False
    )


@dataclasses.dataclass
class UIToolInputAvailableEvent:
    """Indicates that tool input is complete and ready for execution."""

    tool_call_id: str
    tool_name: str
    input: Any
    type: Literal["tool-input-available"] = dataclasses.field(
        default="tool-input-available", init=False
    )
    provider_executed: bool | None = None
    provider_metadata: dict[str, Any] | None = None
    tool_metadata: dict[str, Any] | None = None
    dynamic: bool | None = None
    title: str | None = None


@dataclasses.dataclass
class UIToolInputErrorEvent:
    """Indicates an error occurred during tool input processing."""

    tool_call_id: str
    tool_name: str
    input: Any
    error_text: str
    type: Literal["tool-input-error"] = dataclasses.field(
        default="tool-input-error", init=False
    )
    provider_executed: bool | None = None
    provider_metadata: dict[str, Any] | None = None
    tool_metadata: dict[str, Any] | None = None
    dynamic: bool | None = None
    title: str | None = None


@dataclasses.dataclass
class UIToolOutputAvailableEvent:
    """Contains the result of tool execution."""

    tool_call_id: str
    output: Any
    type: Literal["tool-output-available"] = dataclasses.field(
        default="tool-output-available", init=False
    )
    provider_executed: bool | None = None
    provider_metadata: dict[str, Any] | None = None
    tool_metadata: dict[str, Any] | None = None
    dynamic: bool | None = None
    preliminary: bool | None = None


@dataclasses.dataclass
class UIToolOutputErrorEvent:
    """Indicates an error occurred during tool execution."""

    tool_call_id: str
    error_text: str
    type: Literal["tool-output-error"] = dataclasses.field(
        default="tool-output-error", init=False
    )
    provider_executed: bool | None = None
    provider_metadata: dict[str, Any] | None = None
    tool_metadata: dict[str, Any] | None = None
    dynamic: bool | None = None


@dataclasses.dataclass
class UIToolOutputDeniedEvent:
    """Indicates tool execution was denied."""

    tool_call_id: str
    type: Literal["tool-output-denied"] = dataclasses.field(
        default="tool-output-denied", init=False
    )


@dataclasses.dataclass
class UIToolApprovalRequestEvent:
    """Requests approval for tool execution."""

    approval_id: str
    tool_call_id: str
    type: Literal["tool-approval-request"] = dataclasses.field(
        default="tool-approval-request", init=False
    )
    is_automatic: bool | None = None


@dataclasses.dataclass
class UIToolApprovalResponseEvent:
    """Records an approval decision for a tool call."""

    approval_id: str
    approved: bool
    type: Literal["tool-approval-response"] = dataclasses.field(
        default="tool-approval-response", init=False
    )
    reason: str | None = None
    provider_executed: bool | None = None
    provider_metadata: dict[str, Any] | None = None


@dataclasses.dataclass
class UIStartStepEvent:
    """Indicates the start of a step."""

    type: Literal["start-step"] = dataclasses.field(
        default="start-step", init=False
    )


@dataclasses.dataclass
class UIFinishStepEvent:
    """Indicates that a step has been completed."""

    type: Literal["finish-step"] = dataclasses.field(
        default="finish-step", init=False
    )


@dataclasses.dataclass
class UIFinishEvent:
    """Indicates the completion of a message."""

    type: Literal["finish"] = dataclasses.field(default="finish", init=False)
    finish_reason: FinishReason | None = None
    message_metadata: Any | None = None


@dataclasses.dataclass
class UIAbortEvent:
    """Indicates the message was aborted."""

    type: Literal["abort"] = dataclasses.field(default="abort", init=False)
    reason: str | None = None


@dataclasses.dataclass
class UIMessageMetadataEvent:
    """Contains message metadata."""

    message_metadata: Any
    type: Literal["message-metadata"] = dataclasses.field(
        default="message-metadata", init=False
    )


@dataclasses.dataclass
class UIErrorEvent:
    """Errors appended to the message as they are received."""

    error_text: str
    type: Literal["error"] = dataclasses.field(default="error", init=False)


UIMessageStreamEvent = (
    UIStartEvent
    | UITextStartEvent
    | UITextDeltaEvent
    | UITextEndEvent
    | UIReasoningStartEvent
    | UIReasoningDeltaEvent
    | UIReasoningEndEvent
    | UICustomEvent
    | UISourceUrlEvent
    | UISourceDocumentEvent
    | UIFileEvent
    | UIReasoningFileEvent
    | UIDataEvent
    | UIToolInputStartEvent
    | UIToolInputDeltaEvent
    | UIToolInputAvailableEvent
    | UIToolInputErrorEvent
    | UIToolOutputAvailableEvent
    | UIToolOutputErrorEvent
    | UIToolOutputDeniedEvent
    | UIToolApprovalRequestEvent
    | UIToolApprovalResponseEvent
    | UIStartStepEvent
    | UIFinishStepEvent
    | UIFinishEvent
    | UIAbortEvent
    | UIMessageMetadataEvent
    | UIErrorEvent
)
