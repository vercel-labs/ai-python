"""Speech-to-text via dedicated transcription models.

::

    import ai

    model = ai.get_model("openai/whisper-1")
    audio = pathlib.Path("speech.mp3").read_bytes()

    result = await ai.ops.transcribe(model, audio)
    result.value.text  # the transcript
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .. import types
    from ..models.core import model as model_
    from . import items


class TranscriptionSegment(pydantic.BaseModel):
    """A portion of the transcript with timing information."""

    text: str
    start_second: float
    end_second: float

    model_config = pydantic.ConfigDict(frozen=True)


class Transcription(pydantic.BaseModel):
    """A transcript produced by a transcription model."""

    text: str
    """The complete transcribed text."""
    segments: list[TranscriptionSegment] = []
    """Transcript segments with timing information, when reported."""
    language: str | None = None
    """Detected language as an ISO 639-1 code, e.g. ``"en"``."""
    duration_seconds: float | None = None
    """Total duration of the input audio in seconds, when reported."""

    model_config = pydantic.ConfigDict(frozen=True)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TranscribeParams:
    """Parameters for transcription."""

    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


async def transcribe(
    model: model_.Model,
    audio: types.messages.FilePart | bytes,
    *,
    params: TranscribeParams | None = None,
) -> items.Item[Transcription]:
    """Transcribe audio with a dedicated transcription model.

    ``audio`` is a :class:`~ai.types.messages.FilePart` (URL, base-64
    string, or bytes data) or raw audio bytes. Returns an
    :class:`~ai.ops.Item` whose ``value`` is the transcript.
    Transcription models do not report token usage; cost information,
    when the provider sends it, is on ``.provider_metadata``.
    """
    return await model.provider.transcribe(
        model, audio, params=params or TranscribeParams()
    )
