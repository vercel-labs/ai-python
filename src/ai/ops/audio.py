"""Speech generation via dedicated speech models.

::

    import ai

    model = ai.get_model("openai/tts-1")

    result = await ai.ops.generate_audio(model, "Hello from the AI SDK!")
    result.value  # list[FilePart]
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from .. import experimental_telemetry as telemetry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .. import types
    from ..models.core import model as model_
    from . import items


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioPrompt:
    """Prompt for speech generation: the text plus delivery instructions."""

    text: str
    """The text to convert to speech."""
    instructions: str | None = None
    """Instructions for speech delivery, e.g. tone or emotion."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioParams:
    """Parameters for speech generation."""

    voice: str | None = None
    """Provider-specific voice ID or name."""
    output_format: str | None = None
    """Audio output format, e.g. ``"mp3"`` or ``"wav"``."""
    speed: float | None = None
    """Speech speed multiplier."""
    language: str | None = None
    """Output language as an ISO 639-1 code, e.g. ``"en"``."""
    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


async def generate_audio(
    model: model_.Model,
    prompt: str | AudioPrompt,
    *,
    params: AudioParams | None = None,
) -> items.Item[list[types.messages.FilePart]]:
    """Generate speech audio with a dedicated speech model.

    ``prompt`` is the text to speak, or an :class:`AudioPrompt` carrying
    delivery instructions alongside the text. Returns an
    :class:`~ai.ops.Item` whose ``value`` is the generated audio files.
    Speech models do not report token usage; cost information, when the
    provider sends it, is on ``.provider_metadata``.
    """
    if isinstance(prompt, str):
        prompt = AudioPrompt(text=prompt)
    data = telemetry.GenerateAudioSpanData(
        model=model.id,
        provider=model.provider.name,
    )
    async with telemetry.span(data) as sp:
        item = await model.provider.generate_audio(
            model, prompt, params=params or AudioParams()
        )
        sp.data.usage = item.usage
        sp.data.output_count = len(item.value)
        if item.warnings:
            sp.data.warnings = [w.model_dump() for w in item.warnings]
        return item
