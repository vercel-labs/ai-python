from typing import Any

import pydantic

_PARAMS_CONFIG = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class ImageParams(pydantic.BaseModel):
    """Parameters for image generation (``/image-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    size: str | None = None
    aspect_ratio: str | None = pydantic.Field(
        default=None, serialization_alias="aspectRatio"
    )
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, serialization_alias="providerOptions"
    )


class VideoParams(pydantic.BaseModel):
    """Parameters for video generation (``/video-model`` endpoint)."""

    model_config = _PARAMS_CONFIG

    n: int = 1
    aspect_ratio: str | None = pydantic.Field(
        default=None, serialization_alias="aspectRatio"
    )
    resolution: str | None = None
    duration: int | None = None
    fps: int | None = None
    seed: int | None = None
    provider_options: dict[str, Any] = pydantic.Field(
        default_factory=dict, serialization_alias="providerOptions"
    )


GenerateParams = ImageParams | VideoParams
