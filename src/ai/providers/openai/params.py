from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class OpenAIParams:
    """OpenAI-specific request parameters."""

    store: bool | None = None
    """Whether to store the response for later retrieval."""


__all__ = ["OpenAIParams"]
