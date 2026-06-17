import base64
import uuid
from typing import Annotated, Any, Literal, Self, overload

import pydantic
from pydantic_core import to_jsonable_python

from . import media
from . import usage as usage_


def generate_id(prefix: str | None = None) -> str:
    """Generate a short random ID for messages and parts."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw


class TextPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    text: str
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["text"] = "text"


class FilePart(pydantic.BaseModel):
    """File, image, or audio content part.

    Covers images (``image/*``), documents (``application/pdf``, ``text/*``),
    and audio (``audio/*``).  The ``media_type`` field tells provider
    converters how to format this part for each API.

    ``data`` accepts:

    * **str** -- a URL (``http(s)://...`` or ``data:...``) *or* raw
      base-64 text.
    * **bytes** -- raw binary data (will be base-64 encoded when serialized
      to JSON for providers that need it).
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    data: str | bytes
    media_type: str  # IANA media type, e.g. "image/png", "audio/wav"
    filename: str | None = None
    kind: Literal["file"] = "file"
    provider_metadata: dict[str, Any] | None = None

    @pydantic.field_serializer("data", when_used="json")
    @classmethod
    def _serialize_data(cls, v: str | bytes, _info: Any) -> str:
        """Encode ``bytes`` as standard base-64 for JSON serialization.

        Pydantic's built-in ``ser_json_bytes`` uses URL-safe base-64
        (``-`` and ``_``) which LLM provider APIs reject.  This
        serializer uses standard base-64 (``+`` and ``/``) instead.
        ``str`` values (URLs, existing base-64) pass through unchanged.
        """
        if isinstance(v, bytes):
            return base64.b64encode(v).decode("ascii")
        return v

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
                "Cannot detect media_type from bytes. "
                "Provide media_type explicitly."
            )
        return cls(data=data, media_type=media_type, filename=filename)


# ---------------------------------------------------------------------------
# Multipart tool result -- a tool may return a mix of text and file/image
# parts so the model sees actual media.  Stored on ``ToolResultPart.result``
# with ``result_kind="special"``; providers expand it into their multimodal
# wire format.
# ---------------------------------------------------------------------------


ContentPart = Annotated[
    TextPart | FilePart,
    pydantic.Field(discriminator="kind"),
]


class ContentOutput(pydantic.BaseModel):
    """Multipart tool result -- mix of text and file/image parts."""

    type: Literal["content"] = "content"
    value: list[ContentPart]

    model_config = pydantic.ConfigDict(frozen=True)


class MessageBundle(pydantic.BaseModel):
    type: Literal["messages"] = "messages"
    messages: tuple["Message", ...]


SpecialToolResult = ContentOutput | MessageBundle

_SPECIAL_TOOL_RESULT_ADAPTER: pydantic.TypeAdapter[SpecialToolResult] = (
    pydantic.TypeAdapter(
        Annotated[
            SpecialToolResult,
            pydantic.Field(discriminator="type"),
        ]
    )
)


def _jsonify_result(value: Any) -> Any:
    """Reduce a tool-result value to JSON-y data.

    :class:`ContentOutput` and :class:`MessageBundle` are kept as typed
    models -- providers and the UI adapter dispatch on ``isinstance``.
    Everything else, including any other pydantic model, is dumped to plain
    JSON-y Python so a tool result never carries an arbitrary model and its
    in-memory shape matches what survives a serialization round-trip.
    """
    if isinstance(value, SpecialToolResult):
        return value
    # Raise (rather than stringify) on a value that isn't JSON-serializable:
    # a tool returning such a result is a bug worth surfacing, not hiding.
    return to_jsonable_python(value)


_MODEL_INPUT_UNSET: Any = object()

# Coarse tag for the shape of ``ToolResultPart.result``.
# ``"special"`` means a :class:`SpecialToolResult`; ``"error"`` flags
# an error result; ``"json"`` (the default) is any plain value.
# Providers decide text-vs-json at the wire boundary (a ``str`` is
# sent raw, everything else is JSON-encoded).
ResultKind = Literal["error", "json", "special"]


class ToolResultPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    tool_call_id: str
    tool_name: str
    is_hook_pending: bool = False
    provider_metadata: dict[str, Any] | None = None

    # The "real" result of the tool call.  Stays ``Any``: a plain value
    # (str, dict, BaseModel, ...), a :class:`ContentOutput` for multipart
    # results, or an aggregator snapshot.  ``result_kind`` tags its shape.
    result: Any = None
    result_kind: ResultKind = "json"

    # Value the LLM sees on its next turn.  For most tools this is
    # identical to ``result``; for aggregator-backed tools (sub-agents,
    # streaming-text) it's derived from the aggregator's
    # ``get_model_input``.  Not part of the wire model: it's populated
    # by tool execution and by ``Agent.run`` (which has the tool
    # registry) rather than carried across serialization.  ``default_factory``
    # preserves singleton identity so the unset sentinel survives pydantic's
    # default-copying.
    _model_input: Any = pydantic.PrivateAttr(
        default_factory=lambda: _MODEL_INPUT_UNSET
    )

    kind: Literal["tool_result"] = "tool_result"
    model_config = pydantic.ConfigDict(frozen=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _normalize_result(cls, data: Any) -> Any:
        """Normalize ``result`` to its stored invariant.

        A serialized special result (a dict tagged ``result_kind="special"``)
        is rebuilt into its :class:`ContentOutput` / :class:`MessageBundle`
        model so providers and the UI adapter can rely on ``isinstance``.
        Any other value is reduced to JSON-y data -- a tool result never
        stores an arbitrary pydantic model (see :func:`_jsonify_result`).
        """
        if not isinstance(data, dict) or "result" not in data:
            return data
        result = data["result"]
        if data.get("result_kind") == "special" and isinstance(result, dict):
            return {
                **data,
                "result": _SPECIAL_TOOL_RESULT_ADAPTER.validate_python(result),
            }
        return {**data, "result": _jsonify_result(result)}

    @staticmethod
    def kind_for(result: Any) -> ResultKind:
        """Derive ``result_kind`` for a non-error result value.

        A :data:`SpecialToolResult` is ``"special"``; anything else is
        ``"json"``.  Error results are tagged ``"error"`` by the
        caller, independent of the value.
        """
        return "special" if isinstance(result, SpecialToolResult) else "json"

    @property
    def is_error(self) -> bool:
        """Whether this result represents an error to the model."""
        return self.result_kind == "error"

    def get_model_input(self) -> Any:
        """Return the value the LLM should see, falling back to ``result``."""
        if self._model_input is _MODEL_INPUT_UNSET:
            return self.result
        return self._model_input

    def set_model_input(self, value: Any) -> None:
        """Set the model-facing value (overrides the ``result`` fallback).

        Reduced to JSON-y data like ``result`` so the model never sees an
        arbitrary pydantic model (see :func:`_jsonify_result`).
        """
        self._model_input = _jsonify_result(value)

    @property
    def has_model_input(self) -> bool:
        """Whether ``set_model_input`` has been called on this part."""
        return self._model_input is not _MODEL_INPUT_UNSET


class ToolCallPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    tool_call_id: str
    tool_name: str
    tool_args: str
    provider_metadata: dict[str, Any] | None = None

    # Runtime cache used by replay-from-pending-hook flows: when a prior
    # run completed this tool call but a sibling tool call was suspended
    # on a hook, we fold the completed result onto the ``ToolCallPart``
    # so re-execution short-circuits to the cached value instead of
    # running the tool body again.
    cached_result: ToolResultPart | None = pydantic.Field(
        default=None,
        exclude_if=lambda v: v is None,
        repr=False,
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

    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    tool_call_id: str
    tool_name: str
    tool_args: str = ""
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["builtin_tool_call"] = "builtin_tool_call"


class BuiltinToolReturnPart(pydantic.BaseModel):
    """The provider's result for a :class:`BuiltinToolCallPart`."""

    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["builtin_tool_return"] = "builtin_tool_return"
    model_config = pydantic.ConfigDict(frozen=True)


class ReasoningPart(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    text: str
    provider_metadata: dict[str, Any] | None = None

    kind: Literal["reasoning"] = "reasoning"


class HookPart[T](pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: generate_id("part"))
    hook_id: str
    hook_type: str
    status: Literal["pending", "resolved", "cancelled"]
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    resolution: T | None = None

    kind: Literal["hook"] = "hook"
    model_config = pydantic.ConfigDict(frozen=True)


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
    id: str = pydantic.Field(default_factory=lambda: generate_id("msg"))
    turn_id: str | None = None
    usage: usage_.Usage | None = None
    provider_metadata: dict[str, Any] | None = None

    # Set on the seeded message that ``models.stream`` returns when
    # short-circuiting an existing assistant turn (resume-after-approval
    # flows).  ``Context.add`` skips replay-flagged messages so the loop
    # can call ``context.add(stream.message)`` unconditionally without
    # producing a duplicate turn.
    replay: bool = pydantic.Field(
        default=False,
        exclude_if=lambda v: not v,
        repr=False,
    )

    @property
    def text(self) -> str:
        """Concatenated text parts."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def reasoning(self) -> str:
        """Concatenated reasoning parts."""
        return "".join(
            p.text for p in self.parts if isinstance(p, ReasoningPart)
        )

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
    def get_output(
        self, output_type: type[pydantic.BaseModel] | None = None
    ) -> Any:
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
                f"got role={self.role!r} with "
                f"{len(self.tool_calls)} tool call(s)"
            )
        if output_type is None:
            return self.text
        return output_type.model_validate_json(self.text)


# ``MessageBundle`` forward-references ``Message``, which is defined later in
# this module, so its schema (and the adapter built from it) is incomplete at
# class-creation time.  Rebuild both once ``Message`` exists so importers never
# see a half-built model.
MessageBundle.model_rebuild()
_SPECIAL_TOOL_RESULT_ADAPTER.rebuild(force=True)
