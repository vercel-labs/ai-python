from typing import Annotated, Literal

import pydantic

from . import messages

# we're using pydantic because events are crossing
# serialization border in the case of durable execution


class Start(pydantic.BaseModel):
    kind: Literal["start"] = "start"
    model_config = pydantic.ConfigDict(frozen=True)


class End(pydantic.BaseModel):
    kind: Literal["end"] = "end"
    model_config = pydantic.ConfigDict(frozen=True)


class MessageStart(pydantic.BaseModel):
    message: messages.Message

    kind: Literal["message_start"] = "message_start"
    model_config = pydantic.ConfigDict(frozen=True)


class MessageEnd(pydantic.BaseModel):
    message: messages.Message

    kind: Literal["message_end"] = "message_end"
    model_config = pydantic.ConfigDict(frozen=True)


class PartStart(pydantic.BaseModel):
    part: messages.Part

    kind: Literal["part_start"] = "part_start"
    model_config = pydantic.ConfigDict(frozen=True)


class PartDelta(pydantic.BaseModel):
    part: messages.Part
    chunk: str

    kind: Literal["part_delta"] = "part_delta"
    model_config = pydantic.ConfigDict(frozen=True)


class PartEnd(pydantic.BaseModel):
    part: messages.Part

    kind: Literal["part_end"] = "part_end"
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
    | PartStart
    | PartDelta
    | PartEnd
    | HookSuspention
    | HookResolution,
    pydantic.Field(discriminator="kind"),
]
