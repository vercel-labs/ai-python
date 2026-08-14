"""Video generation via dedicated video models.

::

    import ai

    model = ai.get_model("google/veo-3.0-generate-001")

    result = await ai.ops.generate_video(model, "A cat walking on a beach")
    result.value  # list[FilePart]
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .. import types
    from ..models.core import model as model_
    from . import items


@dataclasses.dataclass(frozen=True, kw_only=True)
class FrameImage:
    """A role-tagged image input for first/last-frame video generation."""

    image: types.messages.FilePart | bytes | str
    frame_type: Literal["first_frame", "last_frame"]
    """Whether the model animates from this image (``"first_frame"``) or
    towards it (``"last_frame"``)."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class VideoPrompt:
    """Prompt for video generation: text plus optional input media.

    Input files accept a :class:`~ai.types.messages.FilePart`, raw
    ``bytes``, or a ``str`` URL / base-64 data.
    """

    text: str | None = None
    """Text prompt."""
    image: types.messages.FilePart | bytes | str | None = None
    """Input image for image-to-video generation (the starting frame)."""
    frame_images: Sequence[FrameImage] = ()
    """Role-tagged first/last-frame images. A ``first_frame`` entry takes
    precedence over :attr:`image` as the start image."""
    references: Sequence[types.messages.FilePart | bytes | str] = ()
    """Reference images or videos for reference-to-video generation.
    Cannot be combined with :attr:`frame_images`."""


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
    generate_audio: bool | None = None
    """Whether the model should generate audio alongside the video."""
    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


async def generate_video(
    model: model_.Model,
    prompt: str | VideoPrompt,
    *,
    params: VideoParams | None = None,
) -> items.Item[list[types.messages.FilePart]]:
    """Generate videos with a dedicated video model.

    ``prompt`` is the text prompt, or a :class:`VideoPrompt` carrying an
    input image, role-tagged frame images, or reference media alongside
    the text. Returns an :class:`~ai.ops.Item` whose ``value`` is the
    generated videos, with provider warnings (e.g. an ignored parameter)
    on ``.warnings``. Video models do not report token usage; cost
    information, when the provider sends it, is on ``.provider_metadata``.
    """
    if isinstance(prompt, str):
        prompt = VideoPrompt(text=prompt)
    return await model.provider.generate_video(
        model, prompt, params=params or VideoParams()
    )
