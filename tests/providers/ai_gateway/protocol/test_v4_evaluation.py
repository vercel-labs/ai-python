"""AI Gateway v4 evaluation-model operation tests."""

from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import pydantic
import pytest

import ai
from ai import ops

from ..conftest import mock_model

_MODEL_ID = "typesafe-ai/jev"


class Questions(pydantic.BaseModel):
    department: ops.experimental.ChoiceQuestion
    severity: ops.experimental.ScoreQuestion
    refund: ops.experimental.NoulQuestion


class Answers(pydantic.BaseModel):
    department: ops.experimental.ChoiceAnswer
    severity: ops.experimental.ScoreAnswer
    refund: ops.experimental.NoulAnswer


class NoulQuestions(pydantic.BaseModel):
    answer: ops.experimental.NoulQuestion


class NoulAnswers(pydantic.BaseModel):
    answer: ops.experimental.NoulAnswer


def questions() -> Questions:
    return Questions(
        department=ops.experimental.ChoiceQuestion(
            instructions="Which team should handle this?",
            criteria={"billing": "Charges", "support": "Other requests"},
        ),
        severity=ops.experimental.ScoreQuestion(
            instructions="How severe is this?",
            criteria=["Cosmetic", "Workaround exists", "Blocking"],
        ),
        refund=ops.experimental.NoulQuestion(
            instructions="Is a refund requested?",
            criteria={"true": "A refund is requested", "false": None},
        ),
    )


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

    result = await ops.experimental.evaluate(
        mock_model(
            httpx.MockTransport(handler),
            api_key="sk-test",
            model_id=_MODEL_ID,
        ),
        {"message": "Please refund the duplicate charge."},
        questions(),
        output_type=Answers,
        params=ops.experimental.EvaluationParams(
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

    assert isinstance(result.value, Answers)
    assert result.value.department.choice == "billing"
    assert result.value.department.probabilities == {
        "billing": 0.8,
        "support": 0.2,
    }
    assert result.value.severity.score == 1.5
    assert result.value.department.confidence == 0.91
    assert result.value.severity.confidence == 0.87
    assert result.value.refund.noul == 0.98
    assert result.metadata == {
        "rounding": {
            "probabilityDecimals": 2,
            "scoreDecimals": 2,
        }
    }
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


@pytest.mark.parametrize(
    "provider_metadata",
    [
        None,
        {},
        {"typesafe": {}},
        {"typesafe": {"confidence": {"department": 0.0}}},
        {"typesafe": {"confidence": {"department": 1.0, "severity": 0.5}}},
    ],
)
async def test_evaluate_dynamic_confidence_and_noul(
    provider_metadata: dict[str, Any] | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["questions"]["refund"] == {
            "type": "boolean",
            "instructions": "Refund requested?",
        }
        return httpx.Response(
            200,
            json={
                "answers": {
                    "department": {"type": "choice", "choice": "billing"},
                    "severity": {"type": "score", "score": 1.5},
                    "refund": {"type": "boolean", "probability": 0.98},
                },
                "providerMetadata": provider_metadata,
            },
        )

    question_values = questions()
    result = await ops.experimental.evaluate(
        mock_model(httpx.MockTransport(handler), model_id=_MODEL_ID),
        "state",
        {
            "department": question_values.department,
            "severity": question_values.severity,
            "refund": ops.experimental.NoulQuestion(
                instructions="Refund requested?"
            ),
        },
    )
    confidence = (
        (provider_metadata or {}).get("typesafe", {}).get("confidence", {})
    )
    department = result.value["department"]
    severity = result.value["severity"]
    refund = result.value["refund"]
    assert isinstance(department, ops.experimental.ChoiceAnswer)
    assert isinstance(severity, ops.experimental.ScoreAnswer)
    assert isinstance(refund, ops.experimental.NoulAnswer)
    assert department.confidence == confidence.get("department")
    assert severity.confidence == confidence.get("severity")
    assert refund.model_dump() == {"type": "noul", "noul": 0.98}
    assert result.provider_metadata == provider_metadata


@pytest.mark.parametrize("confidence", [-0.1, 1.1, "0.5", True])
async def test_evaluate_rejects_invalid_confidence(confidence: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": {"answer": {"type": "choice", "choice": "yes"}},
                "providerMetadata": {
                    "typesafe": {"confidence": {"answer": confidence}}
                },
            },
        )

    with pytest.raises(pydantic.ValidationError, match="confidence"):
        await ops.experimental.evaluate(
            mock_model(httpx.MockTransport(handler), model_id=_MODEL_ID),
            "state",
            {
                "answer": ops.experimental.ChoiceQuestion(
                    instructions="Choose", criteria={"yes": None, "no": None}
                )
            },
        )


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

    result = await ops.experimental.evaluate(
        mock_model(httpx.MockTransport(handler), model_id=_MODEL_ID),
        "state",
        {
            "answer": ops.experimental.ChoiceQuestion(
                instructions="Choose",
                criteria={"yes": None, "no": None},
            )
        },
    )

    assert isinstance(result.value["answer"], ops.experimental.ChoiceAnswer)
    assert result.value["answer"].choice == "yes"
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
        await ops.experimental.evaluate(
            mock_model(httpx.MockTransport(handler), model_id=_MODEL_ID),
            "state",
            NoulQuestions(
                answer=ops.experimental.NoulQuestion(
                    instructions="Is this true?"
                )
            ),
            output_type=NoulAnswers,
        )
