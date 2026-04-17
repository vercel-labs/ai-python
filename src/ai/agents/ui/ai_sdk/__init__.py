"""AI SDK UI adapter — ``ai.Messages`` in, ``ai.Messages`` out, SSE on the wire."""

from .inbound import (
    ApprovalResponse,
    apply_approvals,
    extract_approvals,
    to_messages,
)
from .outbound import to_sse, to_stream, to_ui_messages
from .protocol import UI_MESSAGE_STREAM_HEADERS
from .ui_message import UIMessage

__all__ = [
    "ApprovalResponse",
    "UIMessage",
    "UI_MESSAGE_STREAM_HEADERS",
    "apply_approvals",
    "extract_approvals",
    "to_messages",
    "to_sse",
    "to_stream",
    "to_ui_messages",
]
