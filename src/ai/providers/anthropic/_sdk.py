"""Lazy Anthropic SDK imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from .. import _optional

if TYPE_CHECKING:
    import anthropic


class AnthropicSDK(Protocol):
    AsyncAnthropic: type[anthropic.AsyncAnthropic]
    AnthropicError: type[anthropic.AnthropicError]
    APIConnectionError: type[anthropic.APIConnectionError]
    APIError: type[anthropic.APIError]
    APIResponseValidationError: type[anthropic.APIResponseValidationError]
    APIStatusError: type[anthropic.APIStatusError]
    APITimeoutError: type[anthropic.APITimeoutError]


def import_sdk(*, provider: str = "anthropic") -> AnthropicSDK:
    return cast(
        AnthropicSDK,
        _optional.import_optional_sdk(
            "anthropic",
            provider=provider,
            extra="anthropic",
        ),
    )
