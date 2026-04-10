"""Tests for the gateway error factory.

The factory ``create_gateway_error`` is the real point of contact:
it parses the JSON error response from the gateway server and
dispatches to the correct error class.  These tests use payloads
matching the actual gateway wire format.
"""

from __future__ import annotations

import json

from ai.models.ai_gateway import errors


class TestGatewayErrorBase:
    """Base class behaviour that all concrete errors inherit."""

    def test_generation_id_in_message(self) -> None:
        err = errors.GatewayInternalServerError("boom", generation_id="gen-123")
        assert "[gen-123]" in str(err)
        assert err.generation_id == "gen-123"


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
        err = errors.create_gateway_error(
            response_body=body,
            status_code=401,
            api_key_provided=True,
        )
        assert isinstance(err, errors.GatewayAuthenticationError)
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
        err = errors.create_gateway_error(response_body=body, status_code=400)
        assert isinstance(err, errors.GatewayInvalidRequestError)
        assert err.status_code == 400

    def test_rate_limit_error(self) -> None:
        body = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_exceeded",
            }
        }
        err = errors.create_gateway_error(response_body=body, status_code=429)
        assert isinstance(err, errors.GatewayRateLimitError)

    def test_model_not_found_extracts_model_id(self) -> None:
        body = {
            "error": {
                "message": "Model xyz not found",
                "type": "model_not_found",
                "param": {"modelId": "xyz"},
            }
        }
        err = errors.create_gateway_error(response_body=body, status_code=404)
        assert isinstance(err, errors.GatewayModelNotFoundError)
        assert err.model_id == "xyz"

    def test_model_not_found_without_param(self) -> None:
        body = {
            "error": {
                "message": "Not found",
                "type": "model_not_found",
            }
        }
        err = errors.create_gateway_error(response_body=body, status_code=404)
        assert isinstance(err, errors.GatewayModelNotFoundError)
        assert err.model_id is None

    def test_internal_server_error(self) -> None:
        body = {
            "error": {
                "message": "Database down",
                "type": "internal_server_error",
            }
        }
        err = errors.create_gateway_error(response_body=body, status_code=500)
        assert isinstance(err, errors.GatewayInternalServerError)

    def test_unknown_type_falls_back_to_internal(self) -> None:
        body = {
            "error": {
                "message": "Something weird",
                "type": "alien_error",
            }
        }
        err = errors.create_gateway_error(response_body=body, status_code=500)
        assert isinstance(err, errors.GatewayInternalServerError)

    def test_malformed_json_string(self) -> None:
        err = errors.create_gateway_error(response_body="Not JSON", status_code=500)
        assert isinstance(err, errors.GatewayResponseError)

    def test_missing_error_field(self) -> None:
        body = {"ferror": {"message": "oops"}}
        err = errors.create_gateway_error(response_body=body, status_code=404)
        assert isinstance(err, errors.GatewayResponseError)

    def test_generation_id_extracted(self) -> None:
        body = {
            "error": {
                "message": "Rate limit",
                "type": "rate_limit_exceeded",
            },
            "generationId": "gen-abc",
        }
        err = errors.create_gateway_error(response_body=body, status_code=429)
        assert err.generation_id == "gen-abc"
