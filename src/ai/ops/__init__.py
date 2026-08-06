"""Model operations beyond LLM chat: media generation and friends."""

from .audio import AudioParams, generate_audio
from .images import ImageParams, generate_image
from .videos import VideoParams, generate_video

__all__ = [
    "AudioParams",
    "ImageParams",
    "VideoParams",
    "generate_audio",
    "generate_image",
    "generate_video",
]
