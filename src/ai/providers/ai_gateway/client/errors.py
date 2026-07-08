"""Vercel AI Gateway client error hierarchy.

Maps HTTP error responses from the gateway server to typed Python exceptions.
Each error class corresponds to a specific ``error.type`` value in the
gateway's JSON error response format::

    {
      "error": {
        "message": "...",
        "type": "authentication_error" | "invalid_request_error" | ...,
        "param": ...,
        "code": ...
      },
      "generationId": "..."
    }
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from ._client import AuthMethod

_KEY_URL = "https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%2Fapi-keys"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GatewayError(Exception):
    """Base class for all Vercel AI Gateway client errors."""

    type: str | None = "gateway_error"
    status_code: int
    generation_id: str | None
    is_retryable: bool
    response_body: Any = None
    """Full HTTP error response body from the gateway (parsed JSON when
    possible, else the raw text).

    The gateway includes provider routing/fallback details here -- which
    backends it tried and how each one failed -- so the whole body is
    retained rather than just the extracted ``message``/``type``.
    """

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int = 500,
        generation_id: str | None = None,
        is_retryable: bool | None = None,
        response_body: Any = None,
    ) -> None:
        display = f"{message} [{generation_id}]" if generation_id else message
        super().__init__(display)
        self.message = display
        self.status_code = status_code
        self.generation_id = generation_id
        self.response_body = response_body
        self.is_retryable = (
            status_code in {408, 409, 429} or status_code >= 500
            if is_retryable is None
            else is_retryable
        )


# ---------------------------------------------------------------------------
# Concrete errors — thin subclasses that set type + default status_code
# ---------------------------------------------------------------------------


class GatewayAuthenticationError(GatewayError):
    """Authentication failed (HTTP 401)."""

    type: str | None = "authentication_error"

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        status_code: int = 401,
        generation_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            generation_id=generation_id,
        )

    @classmethod
    def create_contextual(
        cls,
        *,
        auth_method: AuthMethod | None,
        status_code: int = 401,
        generation_id: str | None = None,
    ) -> Self:
        """Build a helpful message based on which auth method was used."""
        if auth_method == "api-key":
            msg = (
                "AI Gateway authentication failed: Invalid API key.\n\n"
                f"Create a new API key: {_KEY_URL}\n\n"
                "Provide via 'api_key' option or "
                "'AI_GATEWAY_API_KEY' environment variable."
            )
        elif auth_method == "oidc":
            msg = (
                "AI Gateway authentication failed: Invalid OIDC token.\n\n"
                "Check that Vercel OIDC is enabled for this project and "
                "that the token has not expired."
            )
        else:
            msg = (
                "AI Gateway authentication failed: "
                "No authentication provided.\n\n"
                f"Create an API key: {_KEY_URL}\n"
                "Provide via 'api_key' option or "
                "'AI_GATEWAY_API_KEY' environment variable."
            )
        return cls(
            msg,
            status_code=status_code,
            generation_id=generation_id,
        )


class GatewayInvalidRequestError(GatewayError):
    """Malformed or invalid request (HTTP 400)."""

    type: str | None = "invalid_request_error"

    def __init__(
        self,
        message: str = "Invalid request",
        *,
        status_code: int = 400,
        generation_id: str | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            generation_id=generation_id,
            is_retryable=is_retryable,
        )


class GatewayRateLimitError(GatewayError):
    """Rate limit exceeded (HTTP 429)."""

    type: str | None = "rate_limit_exceeded"

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        status_code: int = 429,
        generation_id: str | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            generation_id=generation_id,
            is_retryable=is_retryable,
        )


class GatewayModelNotFoundError(GatewayError):
    """Requested model was not found (HTTP 404)."""

    type: str | None = "model_not_found"

    def __init__(
        self,
        message: str = "Model not found",
        *,
        status_code: int = 404,
        model_id: str | None = None,
        generation_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            generation_id=generation_id,
        )
        self.model_id = model_id


class GatewayInternalServerError(GatewayError):
    """Internal error on the gateway server (HTTP 500)."""

    type: str | None = "internal_server_error"

    def __init__(
        self,
        message: str = "Internal server error",
        *,
        status_code: int = 500,
        generation_id: str | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            generation_id=generation_id,
            is_retryable=is_retryable,
        )


class GatewayResponseError(GatewayError):
    """Malformed or unparseable response (HTTP 502)."""

    type: str | None = "response_error"

    def __init__(
        self,
        message: str = "Invalid response",
        *,
        status_code: int = 502,
        response_body: Any = None,
        validation_error: Any = None,
        generation_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            generation_id=generation_id,
            response_body=response_body,
        )
        self.validation_error = validation_error


class GatewayTimeoutError(GatewayError):
    """Gateway request timed out (HTTP 408)."""

    type: str | None = "timeout_error"

    def __init__(
        self,
        message: str = "Request timed out",
        *,
        status_code: int = 408,
        generation_id: str | None = None,
        is_retryable: bool | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            generation_id=generation_id,
            is_retryable=is_retryable,
        )


# ---------------------------------------------------------------------------
# Error factory
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type[GatewayError]] = {
    "authentication_error": GatewayAuthenticationError,
    "invalid_request_error": GatewayInvalidRequestError,
    "rate_limit_exceeded": GatewayRateLimitError,
    "model_not_found": GatewayModelNotFoundError,
    "internal_server_error": GatewayInternalServerError,
}

_MALFORMED = "Invalid error response format: Gateway request failed"


def create_gateway_error(
    *,
    response_body: Any,
    status_code: int,
    auth_method: AuthMethod | None = None,
) -> GatewayError:
    """Create a typed error from a gateway JSON error response.

    Falls back to :class:`GatewayResponseError` when the body doesn't
    match the expected ``{"error": {"message": ..., "type": ...}}``
    shape.
    """
    # Parse the response body
    body: Any = response_body
    if isinstance(body, str | bytes):
        try:
            body = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return GatewayResponseError(
                message=_MALFORMED,
                status_code=status_code,
                response_body=response_body,
                validation_error="Response body is not valid JSON",
            )

    # Validate shape
    error_obj = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error_obj, dict) or "message" not in error_obj:
        reason = (
            "Missing 'error' field in response"
            if not isinstance(error_obj, dict)
            else "Missing 'message' field in error object"
        )
        return GatewayResponseError(
            message=_MALFORMED,
            status_code=status_code,
            response_body=body,
            validation_error=reason,
        )

    message: str = error_obj["message"]
    error_type: str | None = error_obj.get("type")
    generation_id: str | None = body.get("generationId")

    err: GatewayError
    match error_type:
        case "authentication_error":
            err = GatewayAuthenticationError.create_contextual(
                auth_method=auth_method,
                status_code=status_code,
                generation_id=generation_id,
            )

        case "model_not_found":
            param = error_obj.get("param")
            model_id = param.get("modelId") if isinstance(param, dict) else None
            err = GatewayModelNotFoundError(
                message=message,
                status_code=status_code,
                model_id=model_id,
                generation_id=generation_id,
            )

        case _:
            cls = _TYPE_MAP.get(error_type or "", GatewayInternalServerError)
            err = cls(
                message=message,
                status_code=status_code,
                generation_id=generation_id,
            )

    # Retain the full body (provider routing/fallback details live alongside
    # the extracted message/type) for surfacing to callers.
    err.response_body = body
    return err
