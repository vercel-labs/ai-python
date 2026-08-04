"""Video generation via dedicated video models.

::

    import ai

    model = ai.get_model("google/veo-3.0-generate-001")
    msgs = [ai.user_message("A cat walking on a beach")]

    message = await ai.ops.generate_video(model, msgs)
    message.videos  # list[FilePart]
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .. import types
    from ..models.core import model as model_


@dataclasses.dataclass(frozen=True, kw_only=True)
class VideoParams:
    """Parameters for video generation."""

    n: int = 1
    """Number of videos to generate."""
    aspect_ratio: str | None = None
    """Aspect ratio as ``"{width}:{height}"``, e.g. ``"16:9"``."""
    resolution: str | None = None
    """Resolution as ``"{width}x{height}"``, e.g. ``"1920x1080"``."""
    duration: int | None = None
    """Video duration in seconds."""
    fps: int | None = None
    """Frames per second."""
    seed: int | None = None
    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


async def generate_video(
    model: model_.Model,
    messages: list[types.messages.Message],
    *,
    params: VideoParams | None = None,
) -> types.messages.Message:
    """Generate videos with a dedicated video model.

    The prompt is the text of the user/system messages; an input image
    for image-to-video rides along as a ``FilePart`` part in a user
    message. Returns a complete assistant message whose parts are the
    generated videos (``message.videos``). Video models do not report
    token usage; cost information, when the provider sends it, is on
    ``message.provider_metadata``.
    """
    return await model.provider.generate_video(
        model, messages, params=params or VideoParams()
    )
