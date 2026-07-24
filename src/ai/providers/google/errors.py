"""Google GenAI SDK error mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import httpx

from ... import errors as ai_errors

if TYPE_CHECKING:
    from google.genai import errors as genai_errors


def map_error(
    exc: genai_errors.APIError,
    *,
    provider: str | None = None,
    model_id: str | None = None,
) -> ai_errors.ProviderAPIError:
    """Map a Google GenAI SDK exception to the public provider hierarchy."""
    status_code = exc.code if isinstance(exc.code, int) else None
    if status_code == 404 and model_id is not None:
        cls: type[ai_errors.ProviderAPIError] = (
            ai_errors.ProviderModelNotFoundError
        )
    elif status_code is not None:
        cls = ai_errors.http_status_to_provider_status_error_class(status_code)
    else:
        cls = ai_errors.ProviderAPIError
    return _provider_error(cls, exc, provider=provider, model_id=model_id)


def _provider_error(
    cls: type[ai_errors.ProviderAPIError],
    exc: genai_errors.APIError,
    *,
    provider: str | None,
    model_id: str | None,
) -> ai_errors.ProviderAPIError:
    body = exc.details
    if issubclass(cls, ai_errors.ProviderModelNotFoundError):
        if model_id is None:  # pragma: no cover - guarded by map_error
            raise RuntimeError(
                "model_id is required for ProviderModelNotFoundError"
            )
        return cls(
            _message(exc),
            model_id=model_id,
            provider=provider,
            http_context=_http_context(exc),
            body=body,
            error_type=exc.status,
        )
    return cls(
        _message(exc),
        provider=provider,
        http_context=_http_context(exc),
        body=body,
        error_type=exc.status,
    )


def map_httpx_error(
    exc: httpx.HTTPError,
    *,
    provider: str | None = None,
) -> ai_errors.ProviderAPIError:
    """Map a raw httpx transport error to the public provider hierarchy.

    The Google GenAI SDK does not wrap transport failures, so connection
    and timeout errors surface as bare httpx exceptions.
    """
    cls: type[ai_errors.ProviderAPIError] = (
        ai_errors.ProviderTimeoutError
        if isinstance(exc, httpx.TimeoutException)
        else ai_errors.ProviderConnectionError
    )
    return cls(
        str(exc) or type(exc).__name__,
        provider=provider,
        is_retryable=True,
    )


def _http_context(
    exc: genai_errors.APIError,
) -> ai_errors.HTTPErrorContext | None:
    if not isinstance(exc.code, int):
        return None
    response = cast("Any", getattr(exc, "response", None))
    if not isinstance(response, httpx.Response):
        return ai_errors.HTTPErrorContext(status_code=exc.code)
    try:
        request = response.request
    except RuntimeError:
        request = None
    return ai_errors.HTTPErrorContext(
        status_code=exc.code,
        request=request,
        response=response,
    )


def _message(exc: genai_errors.APIError) -> str:
    if isinstance(exc.message, str) and exc.message:
        return exc.message
    return str(exc)


__all__ = ["map_error", "map_httpx_error"]
