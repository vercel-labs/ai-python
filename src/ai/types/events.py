from typing import Annotated, Literal

import pydantic

from . import messages
from . import usage as usage_

# we're using pydantic because events are crossing
# serialization border in the case of durable execution


class Start(pydantic.BaseModel):
    kind: Literal["start"] = "start"
    model_config = pydantic.ConfigDict(frozen=True)


class End(pydantic.BaseModel):
    kind: Literal["end"] = "end"
    model_config = pydantic.ConfigDict(frozen=True)


class MessageStart(pydantic.BaseModel):
    message: messages.Message | None = None

    kind: Literal["message_start"] = "message_start"
    model_config = pydantic.ConfigDict(frozen=True)


class MessageEnd(pydantic.BaseModel):
    message: messages.Message
    usage: usage_.Usage | None = None

    kind: Literal["message_end"] = "message_end"
    model_config = pydantic.ConfigDict(frozen=True)


class TextStart(pydantic.BaseModel):
    block_id: str = ""

    kind: Literal["text_start"] = "text_start"
    model_config = pydantic.ConfigDict(frozen=True)


class TextDelta(pydantic.BaseModel):
    chunk: str
    block_id: str = ""

    kind: Literal["text_delta"] = "text_delta"
    model_config = pydantic.ConfigDict(frozen=True)


class TextEnd(pydantic.BaseModel):
    block_id: str = ""

    kind: Literal["text_end"] = "text_end"
    model_config = pydantic.ConfigDict(frozen=True)


class ReasoningStart(pydantic.BaseModel):
    block_id: str = ""

    kind: Literal["reasoning_start"] = "reasoning_start"
    model_config = pydantic.ConfigDict(frozen=True)


class ReasoningDelta(pydantic.BaseModel):
    chunk: str
    block_id: str = ""

    kind: Literal["reasoning_delta"] = "reasoning_delta"
    model_config = pydantic.ConfigDict(frozen=True)


class ReasoningEnd(pydantic.BaseModel):
    block_id: str = ""
    signature: str | None = None

    kind: Literal["reasoning_end"] = "reasoning_end"
    model_config = pydantic.ConfigDict(frozen=True)


class ToolStart(pydantic.BaseModel):
    tool_call_id: str = ""
    tool_name: str = ""

    kind: Literal["tool_start"] = "tool_start"
    model_config = pydantic.ConfigDict(frozen=True)


class ToolDelta(pydantic.BaseModel):
    chunk: str
    tool_call_id: str = ""

    kind: Literal["tool_delta"] = "tool_delta"
    model_config = pydantic.ConfigDict(frozen=True)


class ToolEnd(pydantic.BaseModel):
    tool_call_id: str = ""

    kind: Literal["tool_end"] = "tool_end"
    model_config = pydantic.ConfigDict(frozen=True)


class HookSuspention(pydantic.BaseModel):
    kind: Literal["hook_suspention"] = "hook_suspention"
    model_config = pydantic.ConfigDict(frozen=True)


class HookResolution(pydantic.BaseModel):
    kind: Literal["hook_resolution"] = "hook_resolution"
    model_config = pydantic.ConfigDict(frozen=True)


Event = Annotated[
    Start
    | End
    | MessageStart
    | MessageEnd
    | TextStart
    | TextDelta
    | TextEnd
    | ReasoningStart
    | ReasoningDelta
    | ReasoningEnd
    | ToolStart
    | ToolDelta
    | ToolEnd
    | HookSuspention
    | HookResolution,
    pydantic.Field(discriminator="kind"),
]
