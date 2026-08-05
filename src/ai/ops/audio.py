"""Speech generation via dedicated speech models.

::

    import ai

    model = ai.get_model("openai/tts-1")
    msgs = [ai.user_message("Hello from the AI SDK!")]

    message = await ai.ops.generate_audio(model, msgs)
    message.audio  # list[FilePart]
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .. import types
    from ..models.core import model as model_


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioParams:
    """Parameters for speech generation."""

    voice: str | None = None
    """Provider-specific voice ID or name."""
    output_format: str | None = None
    """Audio output format, e.g. ``"mp3"`` or ``"wav"``."""
    instructions: str | None = None
    """Instructions for speech delivery, e.g. tone or emotion."""
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
    messages: list[types.messages.Message],
    *,
    params: AudioParams | None = None,
) -> types.messages.Message:
    """Generate speech audio with a dedicated speech model.

    The text to speak is the text of the user/system messages. Returns a
    complete assistant message whose parts are the generated audio files
    (``message.audio``). Speech models do not report token usage; cost
    information, when the provider sends it, is on
    ``message.provider_metadata``.
    """
    return await model.provider.generate_audio(
        model, messages, params=params or AudioParams()
    )
