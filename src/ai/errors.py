"""Framework error hierarchy."""

from __future__ import annotations

import dataclasses

import httpx


@dataclasses.dataclass(frozen=True)
class HTTPErrorContext:
    """HTTP transport context captured from a provider API error."""

    status_code: int
    """HTTP response status code."""

    request: httpx.Request | None = None
    """Concrete ``httpx.Request`` for the provider call, when available."""

    response: httpx.Response | None = None
    """Concrete ``httpx.Response`` for the provider call, when available."""


class AIError(Exception):
    """Base class for framework errors."""


class ConfigurationError(AIError):
    """Required SDK configuration is missing or invalid."""


class InstallationError(ConfigurationError):
    """Required optional dependency is not installed."""


class ProviderError(AIError):
    """Base class for errors raised by model providers."""

    message: str
    """Human-readable error message."""

    provider: str | None
    """Provider name or id associated with the failure, when known."""

    def __init__(
        self,
        message: str = "Provider request failed",
        *,
        provider: str | None = None,
    ) -> None:
        """Create a provider error.

        Args:
            message: Human-readable error message.
            provider: Provider name or id associated with the failure, when
                known.
        """
        super().__init__(message)
        self.message = message
        self.provider = provider


class ProviderNotConfiguredError(ProviderError):
    """Provider cannot be used because required configuration is missing."""


class ProviderAPIError(ProviderError):
    """Provider API request failed before a narrower category was known."""

    request_id: str | None
    """Provider-assigned request identifier, when available."""

    http_context: HTTPErrorContext | None
    """HTTP request/response metadata, when available."""

    body: object | None
    """Provider response body decoded or exposed by the upstream SDK."""

    code: str | None
    """Provider-specific machine-readable error code, when available."""

    param: str | None
    """Provider-specific request parameter associated with the error."""

    type: str | None
    """Provider-specific machine-readable error type, when available."""

    is_retryable: bool
    """Whether retrying the same request may reasonably succeed."""

    def __init__(
        self,
        message: str = "Provider request failed",
        *,
        provider: str | None = None,
        request_id: str | None = None,
        http_context: HTTPErrorContext | None = None,
        body: object | None = None,
        code: str | None = None,
        param: str | None = None,
        error_type: str | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        """Create a provider API error.

        Args:
            message: Human-readable error message.
            provider: Provider name or id associated with the failure, when
                known.
            request_id: Provider-assigned request identifier, if the upstream
                SDK exposes one. Header names differ by provider, so this is
                normalized by each provider mapper.
            http_context: HTTP request/response metadata for failures that
                reached an HTTP status code. This is ``None`` for failures
                without a status, such as connection errors and client-side
                timeouts. ``HTTPErrorContext.status_code`` is required whenever
                this context is present.
            body: Provider response body decoded or exposed by the upstream
                SDK. OpenAI-compatible providers usually expose the inner
                ``error`` object; Anthropic-compatible providers usually expose
                the full response object.
            code: Provider-specific machine-readable error code, when
                available.
            param: Provider-specific request parameter associated with the
                error, when available.
            error_type: Provider-specific machine-readable error type, when
                available. This is stored as ``.type`` to mirror upstream APIs.
            is_retryable: Whether retrying the same request may reasonably
                succeed. When omitted, this defaults from
                ``http_context.status_code`` when a status is present; provider
                mappers may override it for transport-level failures.
        """
        super().__init__(message, provider=provider)
        self.request_id = request_id
        self.http_context = http_context
        self.body = body
        self.code = code
        self.param = param
        self.type = error_type
        self.is_retryable = (
            _is_retryable_status(http_context.status_code if http_context else None)
            if is_retryable is None
            else is_retryable
        )


class ProviderConnectionError(ProviderAPIError):
    """Provider request failed because the client could not connect."""


class ProviderTimeoutError(ProviderConnectionError):
    """Provider request timed out."""


class ProviderResponseError(ProviderAPIError):
    """Provider returned a malformed or unexpected response."""


class ProviderStatusError(ProviderAPIError):
    """Provider returned a non-success HTTP status code."""


class ProviderBadRequestError(ProviderStatusError):
    """Provider rejected the request as malformed or invalid (HTTP 400)."""


class ProviderAuthenticationError(ProviderStatusError):
    """Provider rejected the request credentials (HTTP 401)."""


class ProviderPermissionDeniedError(ProviderStatusError):
    """Provider rejected the request permissions (HTTP 403)."""


class ProviderNotFoundError(ProviderStatusError):
    """Provider resource or model was not found (HTTP 404)."""


class ProviderModelNotFoundError(ProviderNotFoundError):
    """Provider reported that a specific model was not found (HTTP 404)."""

    model_id: str | None
    """Model id that was not found, when known."""

    def __init__(
        self,
        message: str = "Model not found",
        *,
        model_id: str | None = None,
        provider: str | None = None,
        request_id: str | None = None,
        http_context: HTTPErrorContext | None = None,
        body: object | None = None,
        code: str | None = None,
        param: str | None = None,
        error_type: str | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        """Create a model-not-found provider API error.

        Args:
            message: Human-readable error message.
            model_id: Model id that was not found, when known.
            provider: Provider name or id associated with the failure, when
                known.
            request_id: Provider-assigned request identifier, if available.
            http_context: HTTP request/response metadata for failures that
                reached an HTTP status code.
            body: Provider response body decoded or exposed by the upstream SDK.
            code: Provider-specific machine-readable error code, when available.
            param: Provider-specific request parameter associated with the
                error, when available.
            error_type: Provider-specific machine-readable error type, when
                available. This is stored as ``.type`` to mirror upstream APIs.
            is_retryable: Whether retrying the same request may reasonably
                succeed. When omitted, this defaults from the HTTP status when
                present.
        """
        super().__init__(
            message,
            provider=provider,
            request_id=request_id,
            http_context=http_context,
            body=body,
            code=code,
            param=param,
            error_type=error_type,
            is_retryable=is_retryable,
        )
        self.model_id = model_id


class ProviderConflictError(ProviderStatusError):
    """Provider request conflicted with current state (HTTP 409)."""


class ProviderRequestTooLargeError(ProviderStatusError):
    """Provider rejected an oversized request (HTTP 413)."""


class ProviderUnprocessableEntityError(ProviderStatusError):
    """Provider rejected a semantically invalid request (HTTP 422)."""


class ProviderRateLimitError(ProviderStatusError):
    """Provider rate limit was exceeded (HTTP 429)."""


class ProviderInternalServerError(ProviderStatusError):
    """Provider returned an internal server error (HTTP 5xx)."""


class ProviderServiceUnavailableError(ProviderStatusError):
    """Provider service is unavailable (HTTP 503)."""


class ProviderDeadlineExceededError(ProviderStatusError):
    """Provider request deadline was exceeded (HTTP 504)."""


class ProviderOverloadedError(ProviderInternalServerError):
    """Provider is overloaded (HTTP 529)."""


class UnsupportedProviderError(AIError):
    """The SDK does not support or recognize this provider yet."""

    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        super().__init__(f"unsupported provider {provider_id!r}")


def http_status_to_provider_status_error_class(
    status_code: int,
) -> type[ProviderStatusError]:
    """Return the provider status error class for an HTTP status code."""
    if status_code == 400:
        return ProviderBadRequestError
    if status_code == 401:
        return ProviderAuthenticationError
    if status_code == 403:
        return ProviderPermissionDeniedError
    if status_code == 404:
        return ProviderNotFoundError
    if status_code == 409:
        return ProviderConflictError
    if status_code == 413:
        return ProviderRequestTooLargeError
    if status_code == 422:
        return ProviderUnprocessableEntityError
    if status_code == 429:
        return ProviderRateLimitError
    if status_code == 503:
        return ProviderServiceUnavailableError
    if status_code == 504:
        return ProviderDeadlineExceededError
    if status_code == 529:
        return ProviderOverloadedError
    if status_code >= 500:
        return ProviderInternalServerError
    return ProviderStatusError


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code is not None and (
        status_code in {408, 409, 429} or status_code >= 500
    )


__all__ = [
    "AIError",
    "ConfigurationError",
    "HTTPErrorContext",
    "InstallationError",
    "ProviderAPIError",
    "ProviderAuthenticationError",
    "ProviderBadRequestError",
    "ProviderConflictError",
    "ProviderConnectionError",
    "ProviderDeadlineExceededError",
    "ProviderError",
    "ProviderInternalServerError",
    "ProviderModelNotFoundError",
    "ProviderNotFoundError",
    "ProviderNotConfiguredError",
    "ProviderOverloadedError",
    "ProviderPermissionDeniedError",
    "ProviderRateLimitError",
    "ProviderRequestTooLargeError",
    "ProviderResponseError",
    "ProviderServiceUnavailableError",
    "ProviderStatusError",
    "ProviderTimeoutError",
    "ProviderUnprocessableEntityError",
    "UnsupportedProviderError",
    "http_status_to_provider_status_error_class",
]
