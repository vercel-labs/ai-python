"""Model-facing tool output for the external CLIs' history stores."""

import datetime
import decimal
import json
import uuid

import pydantic

from ..types import messages


def format_result(
    part: messages.ToolResultPart | messages.BuiltinToolReturnPart,
) -> str:
    """Render a native tool result as text, never falling back on dump errors.

    Multipart output keeps its text and names the media types in place of
    files. A message bundle contributes only its last nonempty assistant text,
    matching the sub-agent's projection without serializing its transcript.
    Ordinary Pydantic models are JSON-encoded only if their fields are plain
    data or known scalars; unsupported wrappers are omitted as a whole.
    """
    if isinstance(part, messages.BuiltinToolReturnPart):
        return (
            part.result
            if isinstance(part.result, str)
            else json.dumps(part.result)
        )

    value = part.get_model_input()
    if isinstance(value, messages.ContentOutput | messages.MessageBundle):
        value = _project(value)
    return (
        value if isinstance(value, str) else json.dumps(value, default=_project)
    )


def _project(value: object) -> object:
    if isinstance(value, messages.ContentOutput):
        return "\n".join(
            item.text
            if isinstance(item, messages.TextPart)
            else f"[file: {item.media_type}]"
            for item in value.value
        )
    if isinstance(value, messages.MessageBundle):
        return next(
            (
                m.text
                for m in reversed(value.messages)
                if m.role == "assistant" and m.text
            ),
            "",
        )
    if isinstance(value, messages.Message | messages.ToolResultPart):
        return "[unsupported structured output omitted]"
    if isinstance(value, pydantic.BaseModel):
        # Pydantic's serializer would expand nested models without calling
        # this hook. Only let it see ordinary data and known scalar types;
        # a nested model could be a transcript containing raw tool results.
        try:
            json.dumps(dict(value), default=_check_scalar)
        except (TypeError, ValueError):
            return "[unsupported structured output omitted]"
        return value.model_dump(mode="json", exclude_computed_fields=True)
    raise TypeError(f"Unsupported handoff output type: {type(value).__name__}")


def _check_scalar(value: object) -> None:
    if not isinstance(
        value,
        datetime.date
        | datetime.time
        | datetime.timedelta
        | decimal.Decimal
        | uuid.UUID,
    ):
        raise TypeError("unsupported structured output field")
