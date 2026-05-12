import uuid
from typing import Annotated, Any, Literal, Self, overload

import pydantic

from . import media
from . import usage as usage_


def generate_id(prefix: str | None = None) -> str:
    """Generate a short random ID for messages and parts."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw


class TextPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    text: str
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["text"] = "text"


class ToolResultPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    is_hook_pending: bool = False
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["tool_result"] = "tool_result"
    model_config = pydantic.ConfigDict(frozen=True)


class ToolCallPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    tool_args: str
    provider_metadata: dict[str, Any] | None = None

    # Runtime cache used by replay-from-pending-hook flows: when a prior
    # run completed this tool call but a sibling tool call was suspended
    # on a hook, we fold the completed result onto the ``ToolCallPart``
    # so re-execution short-circuits to the cached value instead of
    # running the tool body again.  Excluded from JSON; not part of the
    # wire model.
    cached_result: ToolResultPart | None = pydantic.Field(
        default=None, exclude=True, repr=False
    )

    kind: Literal["tool_call"] = "tool_call"


DUMMY_TOOL_CALL = ToolCallPart(
    id="<invalid>", tool_call_id="", tool_name="", tool_args=""
)


class BuiltinToolCallPart(pydantic.BaseModel):
    """A tool call the provider executed itself (e.g. web_search).

    Distinct from :class:`ToolCallPart` — these are not callable by the
    host. Adapters emit them when a model uses a built-in tool.
    """

    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    tool_args: str = ""
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["builtin_tool_call"] = "builtin_tool_call"


class BuiltinToolReturnPart(pydantic.BaseModel):
    """The provider's result for a :class:`BuiltinToolCallPart`."""

    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["builtin_tool_return"] = "builtin_tool_return"
    model_config = pydantic.ConfigDict(frozen=True)


class ReasoningPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    text: str
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["reasoning"] = "reasoning"


class HookPart[T](pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    hook_id: str
    hook_type: str
    status: Literal["pending", "resolved", "cancelled"]
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    resolution: T | None = None

    kind: Literal["hook"] = "hook"
    model_config = pydantic.ConfigDict(frozen=True)


class FilePart(pydantic.BaseModel):
    """File, image, or audio content part.

    Covers images (``image/*``), documents (``application/pdf``, ``text/*``),
    and audio (``audio/*``).  The ``media_type`` field tells provider
    converters how to format this part for each API.

    ``data`` accepts:

    * **str** -- a URL (``http(s)://...`` or ``data:...``) *or* raw base-64 text.
    * **bytes** -- raw binary data (will be base-64 encoded when serialized
      to JSON for providers that need it).
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
    data: str | bytes
    media_type: str  # IANA media type, e.g. "image/png", "audio/wav"
    filename: str | None = None
    kind: Literal["file"] = "file"
    provider_metadata: dict[str, Any] | None = None

    @classmethod
    def from_url(cls, url: str, *, media_type: str | None = None) -> Self:
        """Create from a URL, inferring ``media_type`` from the URL if omitted.

        Inference handles ``data:`` URLs (the media type is embedded in the
        prefix) and ``http(s)://`` URLs (via :func:`mimetypes.guess_type`).
        Raises :class:`ValueError` if inference fails and no explicit
        ``media_type`` is provided.
        """
        if media_type is None:
            media_type = media.infer_media_type(url)
        return cls(data=url, media_type=media_type)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str | None = None,
        filename: str | None = None,
    ) -> Self:
        """Create from raw bytes, detecting ``media_type`` via magic bytes.

        Attempts image detection first, then audio.  Raises
        :class:`ValueError` if no ``media_type`` is provided and
        detection fails.
        """
        if media_type is None:
            media_type = media.detect_image_media_type(
                data
            ) or media.detect_audio_media_type(data)
        if media_type is None:
            raise ValueError(
                "Cannot detect media_type from bytes. Provide media_type explicitly."
            )
        return cls(data=data, media_type=media_type, filename=filename)


Part = Annotated[
    TextPart
    | ToolCallPart
    | ToolResultPart
    | BuiltinToolCallPart
    | BuiltinToolReturnPart
    | ReasoningPart
    | HookPart[Any]
    | FilePart,
    pydantic.Field(discriminator="kind"),
]


class Message(pydantic.BaseModel):
    role: Literal["user", "assistant", "system", "tool", "internal"]
    parts: list[Part]
    id: str = pydantic.Field(default_factory=generate_id)
    turn_id: str | None = None
    usage: usage_.Usage | None = None
    provider_metadata: dict[str, Any] | None = None

    # Set on the seeded message that ``models.stream`` returns when
    # short-circuiting an existing assistant turn (resume-after-approval
    # flows).  ``Context.add`` skips replay-flagged messages so the loop
    # can call ``context.add(stream.message)`` unconditionally without
    # producing a duplicate turn.  Excluded from JSON: control flag,
    # not data.
    replay: bool = pydantic.Field(default=False, exclude=True, repr=False)

    @property
    def text(self) -> str:
        """Concatenated text parts."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def reasoning(self) -> str:
        """Concatenated reasoning parts."""
        return "".join(p.text for p in self.parts if isinstance(p, ReasoningPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    @property
    def tool_results(self) -> list[ToolResultPart]:
        return [p for p in self.parts if isinstance(p, ToolResultPart)]

    @property
    def builtin_tool_calls(self) -> list[BuiltinToolCallPart]:
        return [p for p in self.parts if isinstance(p, BuiltinToolCallPart)]

    @property
    def builtin_tool_returns(self) -> list[BuiltinToolReturnPart]:
        return [p for p in self.parts if isinstance(p, BuiltinToolReturnPart)]

    @property
    def files(self) -> list[FilePart]:
        return [p for p in self.parts if isinstance(p, FilePart)]

    @property
    def images(self) -> list[FilePart]:
        return [p for p in self.files if p.media_type.startswith("image/")]

    @property
    def videos(self) -> list[FilePart]:
        return [p for p in self.files if p.media_type.startswith("video/")]

    @overload
    def get_output(self, output_type: None = None) -> str: ...
    @overload
    def get_output[T: pydantic.BaseModel](self, output_type: type[T]) -> T: ...
    def get_output(self, output_type: type[pydantic.BaseModel] | None = None) -> Any:
        """Return the final output of this assistant turn.

        With no ``output_type``, returns the concatenated text content.
        With a Pydantic model class, validates the text as JSON against
        it and returns the parsed instance.

        Raises :class:`ValueError` unless the message is a *final*
        assistant message: role ``"assistant"`` with no pending tool calls.
        """
        if self.role != "assistant" or self.tool_calls:
            raise ValueError(
                "get_output() requires a final assistant message "
                "(role='assistant' with no tool calls); "
                f"got role={self.role!r} with {len(self.tool_calls)} tool call(s)"
            )
        if output_type is None:
            return self.text
        return output_type.model_validate_json(self.text)
