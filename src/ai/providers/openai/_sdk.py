"""Lazy OpenAI SDK imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from .. import _optional

if TYPE_CHECKING:
    from collections.abc import Callable

    import openai


class OpenAISDK(Protocol):
    AsyncOpenAI: type[openai.AsyncOpenAI]
    OpenAIError: type[openai.OpenAIError]
    APIConnectionError: type[openai.APIConnectionError]
    APIError: type[openai.APIError]
    APIResponseValidationError: type[openai.APIResponseValidationError]
    APIStatusError: type[openai.APIStatusError]
    APITimeoutError: type[openai.APITimeoutError]


class OpenAIPydantic(Protocol):
    to_strict_json_schema: Callable[[Any], dict[str, Any]]


def import_sdk(*, provider: str = "openai") -> OpenAISDK:
    return cast(
        "OpenAISDK",
        _optional.import_optional_sdk(
            "openai", provider=provider, extra="openai"
        ),
    )


def import_pydantic(*, provider: str = "openai") -> OpenAIPydantic:
    return cast(
        "OpenAIPydantic",
        _optional.import_optional_sdk(
            "openai.lib._pydantic",
            provider=provider,
            extra="openai",
        ),
    )
