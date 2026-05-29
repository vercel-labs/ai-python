"""Map AI Gateway client errors to public ai errors."""

from __future__ import annotations

from ... import errors as ai_errors
from .client import errors as client_errors


def map_error(exc: client_errors.GatewayError) -> ai_errors.ProviderAPIError:
    """Map a Gateway client error to the public provider hierarchy."""
    if isinstance(exc, client_errors.GatewayAuthenticationError):
        return _mapped(ai_errors.ProviderAuthenticationError, exc)
    if isinstance(exc, client_errors.GatewayInvalidRequestError):
        return _mapped(ai_errors.ProviderBadRequestError, exc)
    if isinstance(exc, client_errors.GatewayRateLimitError):
        return _mapped(ai_errors.ProviderRateLimitError, exc)
    if isinstance(exc, client_errors.GatewayModelNotFoundError):
        return ai_errors.ProviderModelNotFoundError(
            str(exc),
            model_id=exc.model_id,
            provider="ai-gateway",
            http_context=_http_context(exc),
            body=exc.response_body,
            error_type=exc.type,
            is_retryable=exc.is_retryable,
        )
    if isinstance(exc, client_errors.GatewayInternalServerError):
        return _mapped(ai_errors.ProviderInternalServerError, exc)
    if isinstance(exc, client_errors.GatewayResponseError):
        return _mapped(ai_errors.ProviderResponseError, exc)
    if isinstance(exc, client_errors.GatewayTimeoutError):
        return _mapped(ai_errors.ProviderTimeoutError, exc)
    return _mapped(ai_errors.ProviderAPIError, exc)


def _mapped(
    cls: type[ai_errors.ProviderAPIError],
    exc: client_errors.GatewayError,
) -> ai_errors.ProviderAPIError:
    return cls(
        str(exc),
        provider="ai-gateway",
        http_context=_http_context(exc),
        body=exc.response_body,
        error_type=exc.type,
        is_retryable=exc.is_retryable,
    )


def _http_context(
    exc: client_errors.GatewayError,
) -> ai_errors.HTTPErrorContext:
    return ai_errors.HTTPErrorContext(status_code=exc.status_code)


__all__ = ["map_error"]
