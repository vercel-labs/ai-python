"""What a settled turn hands back."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ...types import messages as messages_
from ...types import usage as usage_


class Result(BaseModel):
    """One settled turn.

    `text` is the agent's FINAL message: what it said after its last tool
    call. The narration before an action — "I'll write the file now" — is
    real and stays in `messages`; it is not glued onto the answer.
    `messages` is the conversation as of settling, in AI SDK shape — feed
    it straight to `ai.ui.ai_sdk`.
    `finish_reason` uses the AI SDK's vocabulary (gen_ai semconv): "stop",
    "length", "tool_call", "content_filter", plus "cancelled" for a turn the
    caller stopped or a deadline settled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    finish_reason: str
    text: str = ""
    messages: list[messages_.Message] = Field(default_factory=list)
    usage: usage_.Usage = Field(default_factory=usage_.Usage)

    output: Any | None = None
    """The validated value when the turn ran with `output_type=`."""

    prompt_sent: str = ""
    """Exactly what the agent received — including any schema annotation the
    SDK added. A prompt the SDK edited must be inspectable, or the
    transcript contains messages the caller never wrote and cannot explain.
    """

    approval_errors: list[str] = Field(default_factory=list)
    """Approval hooks that raised or timed out.

    Those calls were DENIED (an approval gate that fails open is not a gate),
    and the failures are reported here rather than swallowed.
    """

    @property
    def cancelled(self) -> bool:
        return self.finish_reason == "cancelled"
