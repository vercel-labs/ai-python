"""Lazy TypeSafe SDK imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from .. import _optional

if TYPE_CHECKING:
    import typesafe_sdk


class TypeSafeSDK(Protocol):
    AsyncTypeSafeClient: type[typesafe_sdk.AsyncTypeSafeClient]
    TypeSafeError: type[typesafe_sdk.TypeSafeError]
    TypeSafeAPIError: type[typesafe_sdk.TypeSafeAPIError]
    TypeSafeAPIConnectionError: type[typesafe_sdk.TypeSafeAPIConnectionError]
    TypeSafeAPITimeoutError: type[typesafe_sdk.TypeSafeAPITimeoutError]
    TypeSafeAPIResponseValidationError: type[
        typesafe_sdk.TypeSafeAPIResponseValidationError
    ]


def import_sdk(*, provider: str = "typesafe") -> TypeSafeSDK:
    return cast(
        "TypeSafeSDK",
        _optional.import_optional_sdk(
            "typesafe_sdk",
            provider=provider,
            extra="typesafe",
        ),
    )
