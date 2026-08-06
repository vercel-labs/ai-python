"""Model operations beyond LLM chat: media generation and friends."""

from .images import ImageParams, generate_image

__all__ = ["ImageParams", "generate_image"]
