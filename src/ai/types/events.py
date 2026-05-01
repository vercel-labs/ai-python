from typing import Annotated, Literal

import pydantic

from . import messages
from . import usage as usage_

# we're using pydantic because events are crossing
# serialization border in the case of durable execution


# Placeholder so BaseEvent.message is typed as Message (not Message | None).
# Stream.__anext__ stamps the real in-progress message before yielding,
# so consumers never see this value.
_DUMMY_MESSAGE = messages.Message(id="<unset>", role="assistant", parts=[])


class BaseEvent(pydantic.BaseModel):
    """Common fields stamped onto every event by the streaming wrapper.

    ``message`` carries the in-progress (or final) assistant message; the
    streaming layer aggregates parts into it as deltas arrive and stamps
    a reference onto each yielded event. ``usage`` carries the latest
    usage value reported by the provider (latest-wins across the stream).
    """

    message: messages.Message = _DUMMY_MESSAGE
    usage: usage_.Usage | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class StreamStart(BaseEvent):
    kind: Literal["stream_start"] = "stream_start"


class StreamEnd(BaseEvent):
    kind: Literal["stream_end"] = "stream_end"


class TextStart(BaseEvent):
    block_id: str = ""

    kind: Literal["text_start"] = "text_start"


class TextDelta(BaseEvent):
    chunk: str
    block_id: str = ""

    kind: Literal["text_delta"] = "text_delta"


class TextEnd(BaseEvent):
    block_id: str = ""

    kind: Literal["text_end"] = "text_end"


class ReasoningStart(BaseEvent):
    block_id: str = ""

    kind: Literal["reasoning_start"] = "reasoning_start"


class ReasoningDelta(BaseEvent):
    chunk: str
    block_id: str = ""

    kind: Literal["reasoning_delta"] = "reasoning_delta"


class ReasoningEnd(BaseEvent):
    block_id: str = ""
    signature: str | None = None

    kind: Literal["reasoning_end"] = "reasoning_end"


class ToolStart(BaseEvent):
    tool_call_id: str = ""
    tool_name: str = ""

    kind: Literal["tool_start"] = "tool_start"


class ToolDelta(BaseEvent):
    chunk: str
    tool_call_id: str = ""

    kind: Literal["tool_delta"] = "tool_delta"


class ToolEnd(BaseEvent):
    tool_call: messages.ToolCallPart
    tool_call_id: str = ""

    kind: Literal["tool_end"] = "tool_end"


class BuiltinToolStart(BaseEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    provider_name: str | None = None

    kind: Literal["builtin_tool_start"] = "builtin_tool_start"


class BuiltinToolDelta(BaseEvent):
    chunk: str
    tool_call_id: str = ""

    kind: Literal["builtin_tool_delta"] = "builtin_tool_delta"


class BuiltinToolEnd(BaseEvent):
    tool_call: messages.BuiltinToolCallPart
    tool_call_id: str = ""

    kind: Literal["builtin_tool_end"] = "builtin_tool_end"


class BuiltinToolResult(BaseEvent):
    """Provider returned a result for a built-in tool call."""

    result: messages.BuiltinToolReturnPart
    tool_call_id: str = ""

    kind: Literal["builtin_tool_result"] = "builtin_tool_result"


class FileEvent(BaseEvent):
    """A complete generated file from the LLM (e.g. inline image from Gemini/GPT)."""

    block_id: str = ""
    media_type: str
    data: str | bytes
    filename: str | None = None

    kind: Literal["file"] = "file"


class HookSuspension(BaseEvent):
    kind: Literal["hook_suspension"] = "hook_suspension"


class HookResolution(BaseEvent):
    kind: Literal["hook_resolution"] = "hook_resolution"


Event = Annotated[
    StreamStart
    | StreamEnd
    | TextStart
    | TextDelta
    | TextEnd
    | ReasoningStart
    | ReasoningDelta
    | ReasoningEnd
    | ToolStart
    | ToolDelta
    | ToolEnd
    | BuiltinToolStart
    | BuiltinToolDelta
    | BuiltinToolEnd
    | BuiltinToolResult
    | FileEvent
    | HookSuspension
    | HookResolution,
    pydantic.Field(discriminator="kind"),
]
