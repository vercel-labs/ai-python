"""AI Gateway v4 evaluation-model operation tests."""

from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import pytest

import ai
from ai import ops

from ..conftest import mock_model

_MODEL_ID = "typesafe-ai/jev"


def questions() -> dict[str, ops.EvaluationQuestion]:
    return {
        "department": ops.ChoiceQuestion(
            instructions="Which team should handle this?",
            criteria={"billing": "Charges", "support": "Other requests"},
        ),
        "severity": ops.ScoreQuestion(
            instructions="How severe is this?",
            criteria=["Cosmetic", "Workaround exists", "Blocking"],
        ),
        "refund": ops.BooleanQuestion(
            instructions="Is a refund requested?",
            criteria={"true": "A refund is requested", "false": None},
        ),
    }


async def test_evaluate_request_and_response() -> None:
    captured_body: dict[str, Any] = {}
    captured_headers: dict[str, str] = {}
    captured_url: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        captured_headers.update(dict(request.headers))
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "answers": {
                    "department": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.8, "support": 0.2},
                    },
                    "severity": {
                        "type": "score",
                        "score": 1.5,
                        "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
                    },
                    "refund": {"type": "boolean", "probability": 0.98},
                },
                "rounding": {
                    "probabilityDecimals": 2,
                    "scoreDecimals": 2,
                },
                "usage": {"inputTokens": 42, "outputTokens": 0},
                "warnings": [
                    {"type": "other", "message": "early access model"}
                ],
                "providerMetadata": {
                    "typesafe": {
                        "confidence": {"department": 0.91, "severity": 0.87}
                    }
                },
            },
        )

    result = await ops.experimental_evaluate(
        mock_model(
            httpx.MockTransport(handler),
            api_key="sk-test",
            model_id=_MODEL_ID,
        ),
        {"message": "Please refund the duplicate charge."},
        questions(),
        params=ops.EvaluationParams(
            provider_options={
                "gateway": {
                    "zeroDataRetention": True,
                    "disallowPromptTraining": True,
                },
                "typesafe": {"effort": "high"},
            }
        ),
    )

    assert captured_url == ["https://gw.test/v4/ai/evaluation-model"]
    assert captured_headers["authorization"] == "Bearer sk-test"
    assert captured_headers["ai-evaluation-model-specification-version"] == "4"
    assert captured_headers["ai-model-id"] == _MODEL_ID
    assert captured_body == {
        "state": {"message": "Please refund the duplicate charge."},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Charges",
                    "support": "Other requests",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is this?",
                "criteria": ["Cosmetic", "Workaround exists", "Blocking"],
            },
            "refund": {
                "type": "boolean",
                "instructions": "Is a refund requested?",
                "criteria": {
                    "true": "A refund is requested",
                    "false": None,
                },
            },
        },
        "providerOptions": {
            "gateway": {
                "zeroDataRetention": True,
                "disallowPromptTraining": True,
            },
            "typesafe": {"effort": "high"},
        },
    }

    department = result.value.answers["department"]
    assert isinstance(department, ops.ChoiceAnswer)
    assert department.choice == "billing"
    assert department.probabilities == {"billing": 0.8, "support": 0.2}
    severity = result.value.answers["severity"]
    assert isinstance(severity, ops.ScoreAnswer)
    assert severity.score == 1.5
    refund = result.value.answers["refund"]
    assert isinstance(refund, ops.BooleanAnswer)
    assert refund.probability == 0.98
    assert result.value.rounding == ops.EvaluationRounding(
        probability_decimals=2,
        score_decimals=2,
    )
    assert result.usage == ai.types.usage.Usage(
        input_tokens=42,
        output_tokens=0,
        raw={"inputTokens": 42, "outputTokens": 0},
    )
    assert result.warnings == [
        ops.Warning(kind="other", message="early access model")
    ]
    assert result.provider_metadata == {
        "typesafe": {"confidence": {"department": 0.91, "severity": 0.87}}
    }


async def test_evaluate_maps_all_warning_types() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {"answer": {"type": "choice", "choice": "yes"}},
                "warnings": [
                    {
                        "type": "unsupported",
                        "feature": "providerOptions.test",
                    },
                    {
                        "type": "compatibility",
                        "feature": "state",
                        "details": "converted",
                    },
                    {
                        "type": "deprecated",
                        "setting": "old",
                        "message": "use new",
                    },
                    {"type": "other", "message": "note"},
                ],
            },
        )

    result = await ops.experimental_evaluate(
        mock_model(httpx.MockTransport(handler), model_id=_MODEL_ID),
        "state",
        {
            "answer": ops.ChoiceQuestion(
                instructions="Choose",
                criteria={"yes": None, "no": None},
            )
        },
    )

    assert result.warnings == [
        ops.Warning(kind="unsupported", feature="providerOptions.test"),
        ops.Warning(kind="compatibility", feature="state", details="converted"),
        ops.Warning(kind="deprecated", setting="old", message="use new"),
        ops.Warning(kind="other", message="note"),
    ]


async def test_evaluate_maps_authentication_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
        )

    with pytest.raises(ai.ProviderAuthenticationError):
        await ops.experimental_evaluate(
            mock_model(httpx.MockTransport(handler), model_id=_MODEL_ID),
            "state",
            {"answer": ops.BooleanQuestion(instructions="Is this true?")},
        )
