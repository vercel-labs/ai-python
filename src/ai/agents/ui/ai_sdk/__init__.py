"""AI SDK UI adapter for messages and SSE streams."""

from .inbound import (
    ApprovalResponse,
    apply_approvals,
    extract_approvals,
    to_messages,
)
from .outbound import to_sse, to_stream, to_ui_messages
from .ui_events import UI_MESSAGE_STREAM_HEADERS
from .ui_messages import UIMessage

__all__ = [
    "UI_MESSAGE_STREAM_HEADERS",
    "ApprovalResponse",
    "UIMessage",
    "apply_approvals",
    "extract_approvals",
    "to_messages",
    "to_sse",
    "to_stream",
    "to_ui_messages",
]
