from __future__ import annotations

import json
from typing import Any

import httpx2
import pydantic
import pytest
import typesafe_sdk

import ai
from ai.ops import experimental as evaluation
from ai.providers.typesafe import TypeSafeProvider


def _model(handler: Any) -> ai.Model:
    # Disable SDK retries so error tests don't back off.
    provider = TypeSafeProvider(
        client=typesafe_sdk.AsyncTypeSafeClient(
            api_key="ts-test",
            retry=typesafe_sdk.RetryPolicy(max_retries=0),
            transport=httpx2.MockTransport(handler),
        ),
    )
    return ai.Model(id="jev-latest", provider=provider)


def _response(answers: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 296, "output_tokens": 20},
    }


class Questions(pydantic.BaseModel):
    urgent: evaluation.NoulQuestion
    department: evaluation.ChoiceQuestion
    frustration: evaluation.ScoreQuestion


class Answers(pydantic.BaseModel):
    urgent: evaluation.NoulAnswer
    department: evaluation.ChoiceAnswer
    frustration: evaluation.ScoreAnswer


async def test_evaluate_sends_native_request_and_parses_answers() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json=_response(
                {
                    "urgent": {"type": "noul", "noul": 0.95},
                    "department": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.88, "technical": 0.12},
                        "confidence": 0.81,
                    },
                    "frustration": {
                        "type": "score",
                        "score": 1.05,
                        "legend": {"0": "Calm", "1": "Frustrated"},
                        "probabilities": {"0": 0.05, "1": 0.95},
                        "confidence": 0.92,
                    },
                }
            ),
        )

    result = await evaluation.evaluate(
        _model(handler),
        "Help! My payouts have been failing for 3 days.",
        Questions(
            urgent=evaluation.NoulQuestion(
                instructions="Does this convey urgency?"
            ),
            department=evaluation.ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments", "technical": None},
            ),
            frustration=evaluation.ScoreQuestion(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated"],
            ),
        ),
        output_type=Answers,
    )

    [request] = requests
    assert request.url == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer ts-test"
    assert json.loads(request.content) == {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?",
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments", "technical": None},
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated"],
            },
        },
    }

    assert result.value.urgent.noul == 0.95
    assert result.value.department.choice == "billing"
    assert result.value.department.confidence == 0.81
    assert result.value.frustration.score == 1.05
    assert result.value.frustration.probabilities == {"0": 0.05, "1": 0.95}
    assert result.usage is not None
    assert result.usage.input_tokens == 296
    assert result.usage.output_tokens == 20
    assert result.provider_metadata == {
        "typesafe": {
            "model": "jev-1.13.0",
            "legends": {"frustration": {"0": "Calm", "1": "Frustrated"}},
        }
    }


async def test_evaluate_sends_noul_criteria_and_provider_options() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200, json=_response({"urgent": {"type": "noul", "noul": 0.1}})
        )

    result = await evaluation.evaluate(
        _model(handler),
        {"message": "fine, thanks"},
        {
            "urgent": evaluation.NoulQuestion(
                instructions="Urgent?",
                criteria={"true": "Time-sensitive", "false": "Not urgent"},
            )
        },
        params=evaluation.EvaluationParams(
            provider_options={"typesafe": {"effort": "high"}}
        ),
    )

    [body] = bodies
    assert body["state"] == {"message": "fine, thanks"}
    assert body["effort"] == "high"
    assert body["questions"]["urgent"]["criteria"] == {
        "true": "Time-sensitive",
        "false": "Not urgent",
    }
    assert result.value == {"urgent": evaluation.NoulAnswer(noul=0.1)}
    assert result.provider_metadata == {"typesafe": {"model": "jev-1.13.0"}}


async def test_evaluate_rejects_non_mapping_provider_options() -> None:
    model = _model(lambda request: httpx2.Response(500))

    with pytest.raises(TypeError, match="must be a mapping"):
        await evaluation.evaluate(
            model,
            "state",
            {"q": evaluation.NoulQuestion(instructions="?")},
            params=evaluation.EvaluationParams(
                provider_options={"typesafe": "high"}
            ),
        )


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, ai.ProviderBadRequestError),
        (401, ai.ProviderAuthenticationError),
        (404, ai.ProviderModelNotFoundError),
        (422, ai.ProviderUnprocessableEntityError),
        (500, ai.ProviderInternalServerError),
    ],
)
async def test_evaluate_maps_status_errors(
    status: int, error_type: type[ai.ProviderAPIError]
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={"detail": "nope"},
            headers={"x-typesafe-request-id": "req_123"},
        )

    with pytest.raises(error_type) as exc_info:
        await evaluation.evaluate(
            _model(handler),
            "state",
            {"q": evaluation.NoulQuestion(instructions="?")},
        )

    exc = exc_info.value
    assert type(exc) is error_type
    assert exc.provider == "typesafe"
    assert exc.request_id == "req_123"
    assert exc.body == {"detail": "nope"}
    assert exc.http_context is not None
    assert exc.http_context.status_code == status


async def test_evaluate_maps_connection_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    with pytest.raises(ai.ProviderConnectionError) as exc_info:
        await evaluation.evaluate(
            _model(handler),
            "state",
            {"q": evaluation.NoulQuestion(instructions="?")},
        )

    assert exc_info.value.is_retryable


async def test_evaluate_maps_invalid_responses() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model": "jev-1.13.0"})

    with pytest.raises(ai.errors.ProviderResponseError) as exc_info:
        await evaluation.evaluate(
            _model(handler),
            "state",
            {"q": evaluation.NoulQuestion(instructions="?")},
        )

    assert not exc_info.value.is_retryable
