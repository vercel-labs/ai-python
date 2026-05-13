"""Framework error hierarchy."""

from __future__ import annotations


class AIError(Exception):
    """Base class for framework errors."""


class ConfigurationError(AIError):
    """Required SDK configuration is missing or invalid."""


class UnsupportedProviderError(AIError):
    """The SDK does not support or recognize this provider yet."""

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        super().__init__(f"unsupported provider {provider_id!r}")


__all__ = ["AIError", "ConfigurationError", "UnsupportedProviderError"]
