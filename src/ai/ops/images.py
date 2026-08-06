"""Image generation via dedicated image models.

::

    import ai

    model = ai.get_model("google/imagen-4.0-generate-001")
    msgs = [ai.user_message("A sunset over Tokyo")]

    message = await ai.ops.generate_image(model, msgs)
    message.images  # list[FilePart]
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .. import types
    from ..models.core import model as model_


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImageParams:
    """Parameters for image generation."""

    n: int = 1
    """Number of images to generate."""
    size: str | None = None
    """Image size as ``"{width}x{height}"``, e.g. ``"1024x1024"``."""
    aspect_ratio: str | None = None
    """Aspect ratio as ``"{width}:{height}"``, e.g. ``"16:9"``."""
    seed: int | None = None
    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


async def generate_image(
    model: model_.Model,
    messages: list[types.messages.Message],
    *,
    params: ImageParams | None = None,
) -> types.messages.Message:
    """Generate images with a dedicated image model.

    The prompt is the text of the user/system messages; input images for
    editing ride along as ``FilePart`` parts in the user messages. Returns a
    complete assistant message whose parts are the generated images
    (``message.images``), with usage on ``message.usage`` when the
    provider reports it.
    """
    return await model.provider.generate_image(
        model, messages, params=params or ImageParams()
    )
