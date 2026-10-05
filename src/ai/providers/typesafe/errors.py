"""TypeSafe SDK error mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import errors as ai_errors
from . import _sdk

if TYPE_CHECKING:
    import typesafe_sdk


def map_error(
    exc: typesafe_sdk.TypeSafeError,
    *,
    provider: str | None = None,
    model_id: str | None = None,
) -> ai_errors.ProviderAPIError:
    """Map a TypeSafe SDK exception to the public provider hierarchy."""
    typesafe = _sdk.import_sdk(provider=provider or "typesafe")
    message = str(exc)

    # Failures without an HTTP response.
    if isinstance(exc, typesafe.TypeSafeAPITimeoutError):
        return ai_errors.ProviderTimeoutError(
            message, provider=provider, is_retryable=True
        )
    if isinstance(exc, typesafe.TypeSafeAPIConnectionError):
        return ai_errors.ProviderConnectionError(
            message, provider=provider, is_retryable=True
        )
    if not isinstance(exc, typesafe.TypeSafeAPIError):
        # Client-side SDK failures, such as an invalid request body.
        return ai_errors.ProviderAPIError(message, provider=provider)

    # Failures with an HTTP response. The SDK exposes the status, body, and
    # headers, but not the underlying httpx request and response.
    http_context = ai_errors.HTTPErrorContext(status_code=exc.status)
    if isinstance(exc, typesafe.TypeSafeAPIResponseValidationError):
        return ai_errors.ProviderResponseError(
            message,
            provider=provider,
            request_id=exc.request_id,
            http_context=http_context,
            body=exc.body,
            param=exc.field_path,
        )
    if exc.status == 404 and model_id is not None:
        return ai_errors.ProviderModelNotFoundError(
            message,
            model_id=model_id,
            provider=provider,
            request_id=exc.request_id,
            http_context=http_context,
            body=exc.body,
        )
    cls = ai_errors.http_status_to_provider_status_error_class(exc.status)
    return cls(
        message,
        provider=provider,
        request_id=exc.request_id,
        http_context=http_context,
        body=exc.body,
    )


__all__ = ["map_error"]
