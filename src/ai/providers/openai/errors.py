"""OpenAI SDK error mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from ... import errors as ai_errors
from . import _sdk

if TYPE_CHECKING:
    import openai

_STATUS_ERROR_MAP: dict[str, type[ai_errors.ProviderAPIError]] = {
    "BadRequestError": ai_errors.ProviderBadRequestError,
    "AuthenticationError": ai_errors.ProviderAuthenticationError,
    "PermissionDeniedError": ai_errors.ProviderPermissionDeniedError,
    "NotFoundError": ai_errors.ProviderNotFoundError,
    "ConflictError": ai_errors.ProviderConflictError,
    "UnprocessableEntityError": ai_errors.ProviderUnprocessableEntityError,
    "RateLimitError": ai_errors.ProviderRateLimitError,
    "InternalServerError": ai_errors.ProviderInternalServerError,
}


def map_error(
    exc: openai.OpenAIError,
    *,
    provider: str | None = None,
    model_id: str | None = None,
) -> ai_errors.ProviderAPIError:
    """Map an OpenAI SDK exception to the public provider hierarchy."""
    openai_sdk = _sdk.import_sdk(provider=provider or "openai")
    if isinstance(exc, openai_sdk.APITimeoutError):
        return _provider_error(
            ai_errors.ProviderTimeoutError,
            exc,
            provider=provider,
            model_id=model_id,
            is_retryable=True,
        )
    if isinstance(exc, openai_sdk.APIConnectionError):
        return _provider_error(
            ai_errors.ProviderConnectionError,
            exc,
            provider=provider,
            model_id=model_id,
            is_retryable=True,
        )
    if isinstance(exc, openai_sdk.APIResponseValidationError):
        return _provider_error(
            ai_errors.ProviderResponseError,
            exc,
            provider=provider,
            model_id=model_id,
        )
    if isinstance(exc, openai_sdk.APIStatusError):
        return _map_status_error(
            exc,
            provider=provider,
            model_id=model_id,
        )
    if isinstance(exc, openai_sdk.APIError):
        return _provider_error(
            ai_errors.ProviderAPIError,
            exc,
            provider=provider,
            model_id=model_id,
        )
    return _provider_error(
        ai_errors.ProviderAPIError,
        exc,
        provider=provider,
        model_id=model_id,
    )


def _map_status_error(
    exc: openai.APIStatusError,
    *,
    provider: str | None,
    model_id: str | None,
) -> ai_errors.ProviderAPIError:
    if exc.status_code == 404 and model_id is not None:
        cls: type[ai_errors.ProviderAPIError] = ai_errors.ProviderModelNotFoundError
    else:
        cls = _STATUS_ERROR_MAP.get(
            type(exc).__name__
        ) or ai_errors.http_status_to_provider_status_error_class(exc.status_code)
    return _provider_error(cls, exc, provider=provider, model_id=model_id)


def _provider_error(
    cls: type[ai_errors.ProviderAPIError],
    exc: openai.OpenAIError,
    *,
    provider: str | None,
    model_id: str | None,
    is_retryable: bool | None = None,
) -> ai_errors.ProviderAPIError:
    body = getattr(exc, "body", None)
    if issubclass(cls, ai_errors.ProviderModelNotFoundError):
        if model_id is None:  # pragma: no cover - guarded by _map_status_error
            raise RuntimeError("model_id is required for ProviderModelNotFoundError")
        return cls(
            _message(exc),
            model_id=model_id,
            provider=provider,
            request_id=getattr(exc, "request_id", None),
            http_context=_http_context(exc),
            body=body,
            code=getattr(exc, "code", None),
            param=getattr(exc, "param", None),
            error_type=getattr(exc, "type", None),
            is_retryable=is_retryable,
        )
    return cls(
        _message(exc),
        provider=provider,
        request_id=getattr(exc, "request_id", None),
        http_context=_http_context(exc),
        body=body,
        code=getattr(exc, "code", None),
        param=getattr(exc, "param", None),
        error_type=getattr(exc, "type", None),
        is_retryable=is_retryable,
    )


def _http_context(exc: openai.OpenAIError) -> ai_errors.HTTPErrorContext | None:
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        return None
    request = getattr(exc, "request", None)
    response = getattr(exc, "response", None)
    return ai_errors.HTTPErrorContext(
        status_code=status_code,
        request=request if isinstance(request, httpx.Request) else None,
        response=response if isinstance(response, httpx.Response) else None,
    )


def _message(exc: openai.OpenAIError) -> str:
    message: Any = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(exc)


__all__ = ["map_error"]
