"""Tests for the gateway error factory.

The factory ``create_gateway_error`` is the real point of contact:
it parses the JSON error response from the gateway server and
dispatches to the correct error class.  These tests use payloads
matching the actual gateway wire format.
"""

from __future__ import annotations

import json

import ai
from ai.providers.ai_gateway import errors
from ai.providers.ai_gateway.client import errors as client_errors


class TestGatewayErrorBase:
    """Base class behaviour that all concrete errors inherit."""

    def test_generation_id_in_message(self) -> None:
        err = client_errors.GatewayInternalServerError("boom", generation_id="gen-123")
        assert "[gen-123]" in str(err)
        assert err.generation_id == "gen-123"

    def test_gateway_errors_are_independent(self) -> None:
        assert isinstance(
            client_errors.GatewayAuthenticationError(), client_errors.GatewayError
        )
        assert not isinstance(
            client_errors.GatewayAuthenticationError(), ai.ProviderError
        )
        assert client_errors.GatewayAuthenticationError().status_code == 401

    def test_gateway_errors_map_to_provider_hierarchy(self) -> None:
        assert isinstance(
            errors.map_error(client_errors.GatewayAuthenticationError()),
            ai.ProviderAuthenticationError,
        )
        assert isinstance(
            errors.map_error(client_errors.GatewayInvalidRequestError()),
            ai.ProviderBadRequestError,
        )
        assert isinstance(
            errors.map_error(client_errors.GatewayRateLimitError()),
            ai.ProviderRateLimitError,
        )
        assert isinstance(
            errors.map_error(client_errors.GatewayModelNotFoundError()),
            ai.ProviderModelNotFoundError,
        )
        assert isinstance(
            errors.map_error(client_errors.GatewayInternalServerError()),
            ai.ProviderInternalServerError,
        )
        assert isinstance(
            errors.map_error(client_errors.GatewayResponseError()),
            ai.ProviderResponseError,
        )
        assert isinstance(
            errors.map_error(client_errors.GatewayTimeoutError()),
            ai.ProviderTimeoutError,
        )


class TestCreateGatewayError:
    """The factory must dispatch on ``error.type`` from the response."""

    def test_authentication_error_from_json_string(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            }
        )
        err = client_errors.create_gateway_error(
            response_body=body,
            status_code=401,
            api_key_provided=True,
        )
        assert isinstance(err, client_errors.GatewayAuthenticationError)
        assert err.status_code == 401
        # contextual message includes the key URL
        assert "vercel.com/d?to=" in str(err)

    def test_invalid_request_error(self) -> None:
        body = {
            "error": {
                "message": "Bad format",
                "type": "invalid_request_error",
            }
        }
        err = client_errors.create_gateway_error(response_body=body, status_code=400)
        assert isinstance(err, client_errors.GatewayInvalidRequestError)
        assert err.status_code == 400

    def test_rate_limit_error(self) -> None:
        body = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_exceeded",
            }
        }
        err = client_errors.create_gateway_error(response_body=body, status_code=429)
        assert isinstance(err, client_errors.GatewayRateLimitError)

    def test_model_not_found_extracts_model_id(self) -> None:
        body = {
            "error": {
                "message": "Model xyz not found",
                "type": "model_not_found",
                "param": {"modelId": "xyz"},
            }
        }
        err = client_errors.create_gateway_error(response_body=body, status_code=404)
        assert isinstance(err, client_errors.GatewayModelNotFoundError)
        assert err.model_id == "xyz"

    def test_model_not_found_without_param(self) -> None:
        body = {
            "error": {
                "message": "Not found",
                "type": "model_not_found",
            }
        }
        err = client_errors.create_gateway_error(response_body=body, status_code=404)
        assert isinstance(err, client_errors.GatewayModelNotFoundError)
        assert err.model_id is None

    def test_internal_server_error(self) -> None:
        body = {
            "error": {
                "message": "Database down",
                "type": "internal_server_error",
            }
        }
        err = client_errors.create_gateway_error(response_body=body, status_code=500)
        assert isinstance(err, client_errors.GatewayInternalServerError)

    def test_unknown_type_falls_back_to_internal(self) -> None:
        body = {
            "error": {
                "message": "Something weird",
                "type": "alien_error",
            }
        }
        err = client_errors.create_gateway_error(response_body=body, status_code=500)
        assert isinstance(err, client_errors.GatewayInternalServerError)

    def test_malformed_json_string(self) -> None:
        err = client_errors.create_gateway_error(
            response_body="Not JSON", status_code=500
        )
        assert isinstance(err, client_errors.GatewayResponseError)

    def test_missing_error_field(self) -> None:
        body = {"ferror": {"message": "oops"}}
        err = client_errors.create_gateway_error(response_body=body, status_code=404)
        assert isinstance(err, client_errors.GatewayResponseError)

    def test_generation_id_extracted(self) -> None:
        body = {
            "error": {
                "message": "Rate limit",
                "type": "rate_limit_exceeded",
            },
            "generationId": "gen-abc",
        }
        err = client_errors.create_gateway_error(response_body=body, status_code=429)
        assert err.generation_id == "gen-abc"

    def test_response_error_mapping_preserves_response_body(self) -> None:
        err = client_errors.GatewayResponseError("bad", response_body={"raw": True})
        mapped = errors.map_error(err)
        assert isinstance(mapped, ai.ProviderResponseError)
        assert mapped.body == {"raw": True}
        assert mapped.http_context is not None
        assert mapped.http_context.status_code == 502
