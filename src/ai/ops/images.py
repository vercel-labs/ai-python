"""Image generation via dedicated image models.

::

    import ai

    model = ai.get_model("google/imagen-4.0-generate-001")

    result = await ai.ops.generate_image(model, "A sunset over Tokyo")
    result.value  # list[FilePart]
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .. import types
    from ..models.core import model as model_
    from . import items


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImagePrompt:
    """Prompt for image generation: text plus optional input images.

    Input files accept a :class:`~ai.types.messages.FilePart`, raw
    ``bytes``, or a ``str`` URL / base-64 data.
    """

    text: str | None = None
    """Text prompt. Some operations, like upscaling, may not need one."""
    images: Sequence[types.messages.FilePart | bytes | str] = ()
    """Input images for editing or variation generation."""
    mask: types.messages.FilePart | bytes | str | None = None
    """Mask image for inpainting operations."""


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
    prompt: str | ImagePrompt,
    *,
    params: ImageParams | None = None,
) -> items.Item[list[types.messages.FilePart]]:
    """Generate images with a dedicated image model.

    ``prompt`` is the text prompt, or an :class:`ImagePrompt` carrying
    input images and a mask alongside the text. Returns an
    :class:`~ai.ops.Item` whose ``value`` is the generated images, with
    usage on ``.usage`` when the provider reports it and provider
    warnings (e.g. an ignored parameter) on ``.warnings``.
    """
    if isinstance(prompt, str):
        prompt = ImagePrompt(text=prompt)
    return await model.provider.generate_image(
        model, prompt, params=params or ImageParams()
    )
