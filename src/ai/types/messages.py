import importlib
import uuid
from typing import Annotated, Any, Literal, Self

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

    kind: Literal["text"] = "text"


class ToolCallPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    tool_args: str

    kind: Literal["tool_call"] = "tool_call"


DUMMY_TOOL_CALL = ToolCallPart(
    id="<invalid>", tool_call_id="", tool_name="", tool_args=""
)


class ToolResultPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False

    kind: Literal["tool_result"] = "tool_result"
    model_config = pydantic.ConfigDict(frozen=True)


class BuiltinToolCallPart(pydantic.BaseModel):
    """A tool call the provider executed itself (e.g. web_search).

    Distinct from :class:`ToolCallPart` — these are not callable by the
    host. Adapters emit them when a model uses a built-in tool.
    """

    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    tool_args: str = ""
    provider_name: str | None = None

    kind: Literal["builtin_tool_call"] = "builtin_tool_call"


class BuiltinToolReturnPart(pydantic.BaseModel):
    """The provider's result for a :class:`BuiltinToolCallPart`."""

    id: str = pydantic.Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    provider_name: str | None = None
    provider_details: dict[str, Any] | None = None

    kind: Literal["builtin_tool_return"] = "builtin_tool_return"
    model_config = pydantic.ConfigDict(frozen=True)


class ReasoningPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=generate_id)
    text: str
    # Anthropic's thinking blocks include a signature for cache/verification.
    # This must be preserved and sent back in multi-turn conversations.
    signature: str | None = None

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


# todo redo this structured output situation and simplify it
def _resolve_class(fully_qualified_name: str) -> type[pydantic.BaseModel]:
    """Import and return a class from its fully qualified name.

    E.g. ``"myapp.models.WeatherForecast"`` → the ``WeatherForecast`` class.
    """
    module_path, _, class_name = fully_qualified_name.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Cannot resolve '{fully_qualified_name}': "
            "expected a fully qualified name like 'mypackage.module.ClassName'"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Module '{module_path}' has no attribute '{class_name}'")
    if not (isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)):
        raise TypeError(
            f"'{fully_qualified_name}' is not a pydantic.BaseModel subclass"
        )
    return cls


class StructuredOutputPart(pydantic.BaseModel):
    """Part containing a validated structured output from the LLM.

    ``data`` stores the parsed JSON dict (always serializable).
    ``output_type_name`` stores the fully qualified class name so the typed
    Pydantic model can be lazily rehydrated via the ``value`` property.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=generate_id)
    data: dict[str, Any]
    output_type_name: str
    kind: Literal["structured_output"] = "structured_output"

    _hydrated: Any = pydantic.PrivateAttr(default=None)

    @property
    def value(self) -> Any:
        """Lazily resolve the output type class and validate ``data`` into it."""
        if self._hydrated is None:
            cls = _resolve_class(self.output_type_name)
            self._hydrated = cls.model_validate(self.data)
        return self._hydrated


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
    | StructuredOutputPart
    | FilePart,
    pydantic.Field(discriminator="kind"),
]


class Message(pydantic.BaseModel):
    role: Literal["user", "assistant", "system", "tool", "internal"]
    parts: list[Part]
    id: str = pydantic.Field(default_factory=generate_id)
    turn_id: str | None = None
    source_label: str | None = None
    usage: usage_.Usage | None = None

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

    @property
    def output(self) -> Any:
        """Parsed structured output from the first structured-output part."""
        for part in self.parts:
            if isinstance(part, StructuredOutputPart):
                return part.value
        return None


Usage = usage_.Usage
