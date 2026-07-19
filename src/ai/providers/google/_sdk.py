"""Lazy Google GenAI SDK imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from .. import _optional

if TYPE_CHECKING:
    import google.genai as genai
    from google.genai import errors as genai_errors


class GoogleSDK(Protocol):
    Client: type[genai.Client]


class GoogleErrors(Protocol):
    APIError: type[genai_errors.APIError]
    ClientError: type[genai_errors.ClientError]
    ServerError: type[genai_errors.ServerError]


def import_sdk(*, provider: str = "google") -> GoogleSDK:
    return cast(
        "GoogleSDK",
        _optional.import_optional_sdk(
            "google.genai",
            provider=provider,
            extra="google",
        ),
    )


def import_errors(*, provider: str = "google") -> GoogleErrors:
    return cast(
        "GoogleErrors",
        _optional.import_optional_sdk(
            "google.genai.errors",
            provider=provider,
            extra="google",
        ),
    )
