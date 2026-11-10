from __future__ import annotations

import base64
import json
from typing import Any

import httpx2
import pydantic
import pytest
import typesafe_sdk

import ai
from ai.ops import experimental as decisions
from ai.providers.typesafe import TypeSafeProvider


def _model(handler: Any) -> ai.Model:
    # Disable SDK retries so error tests don't back off, and compression so
    # tests can read large request bodies.
    provider = TypeSafeProvider(
        client=typesafe_sdk.AsyncTypeSafeClient(
            api_key="ts-test",
            retry=typesafe_sdk.RetryPolicy(max_retries=0),
            compression=False,
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
    urgent: decisions.PredicateQuestion
    department: decisions.ChoiceQuestion
    frustration: decisions.ScoreQuestion


class Answers(pydantic.BaseModel):
    urgent: decisions.PredicateAnswer
    department: decisions.ChoiceAnswer
    frustration: decisions.ScoreAnswer


async def test_decide_sends_native_request_and_parses_answers() -> None:
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

    result = await decisions.decide(
        _model(handler),
        "Help! My payouts have been failing for 3 days.",
        Questions(
            urgent=decisions.PredicateQuestion(
                instructions="Does this convey urgency?"
            ),
            department=decisions.ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments", "technical": None},
            ),
            frustration=decisions.ScoreQuestion(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated"],
            ),
        ),
        output_type=Answers,
    )

    [request] = requests
    assert request.url == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer ts-test"
    assert "typesafe-danger-zone-images" not in request.headers
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

    assert result.value.urgent.probability == 0.95
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


async def test_decide_sends_predicate_criteria_and_provider_options() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200, json=_response({"urgent": {"type": "noul", "noul": 0.1}})
        )

    result = await decisions.decide(
        _model(handler),
        {"message": "fine, thanks"},
        {
            "urgent": decisions.PredicateQuestion(
                instructions="Urgent?",
                criteria={"true": "Time-sensitive", "false": "Not urgent"},
            )
        },
        params=decisions.DecideParams(
            provider_options={"typesafe": {"effort": "high"}}
        ),
    )

    [body] = bodies
    assert body["state"] == {"message": "fine, thanks"}
    assert body["effort"] == "high"
    assert body["questions"]["urgent"]["type"] == "noul"
    assert body["questions"]["urgent"]["criteria"] == {
        "true": "Time-sensitive",
        "false": "Not urgent",
    }
    assert result.value == {
        "urgent": decisions.PredicateAnswer(probability=0.1)
    }
    assert result.provider_metadata == {"typesafe": {"model": "jev-1.13.0"}}


_PNG = b"\x89PNG\r\n\x1a\n-photo"
_JPEG = b"\xff\xd8\xff-reference"


def _multipart(request: httpx2.Request) -> tuple[dict[str, Any], list[bytes]]:
    """Return the JSON request part and the asset bodies, in order."""
    boundary = request.headers["content-type"].split("boundary=")[1]
    sections = request.content.split(b"--" + boundary.encode())[1:-1]
    bodies = [section.split(b"\r\n\r\n", 1)[1][:-2] for section in sections]
    return json.loads(bodies[0]), bodies[1:]


async def test_decide_sends_images_as_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []
    downloads: list[str] = []

    async def download(url: str) -> tuple[bytes, str | None]:
        downloads.append(url)
        return _JPEG, "image/jpeg"

    monkeypatch.setattr(ai.models.core.helpers.files, "download", download)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json=_response(
                {
                    "same": {"type": "noul", "noul": 0.9},
                    "coat": {
                        "type": "choice",
                        "choice": "short",
                        "probabilities": {"short": 0.7, "long": 0.3},
                        "confidence": 0.8,
                    },
                }
            ),
        )

    # The same photo appears twice and as base64 text in a data URL.
    photo = ai.file_part(_PNG)
    inline = ai.file_part(
        "data:image/png;base64," + base64.b64encode(_PNG).decode()
    )
    reference = ai.file_part("https://example.com/reference.jpg")

    await decisions.decide(
        _model(handler),
        {"photo": photo, "notes": ["Found today.", photo, inline]},
        {
            "same": decisions.PredicateQuestion(
                instructions=["Is `photo` the same dog as this?", reference],
            ),
            "coat": decisions.ChoiceQuestion(
                instructions="What is the coat like?",
                criteria={"short": reference, "long": None},
            ),
        },
    )

    [request] = requests
    assert request.headers["typesafe-danger-zone-images"] == "true"
    body, assets = _multipart(request)

    # One asset per file part; each reference stands where its file was.
    photo_ref = body["state"]["photo"]
    inline_ref = body["state"]["notes"][2]
    reference_ref = body["questions"]["same"]["instructions"][1]
    assert body["state"]["notes"] == ["Found today.", photo_ref, inline_ref]
    assert body["questions"]["coat"]["criteria"] == {
        "short": reference_ref,
        "long": None,
    }
    assert body["questions"]["same"]["type"] == "noul"
    assert set(body["assets"]) == {photo_ref, inline_ref, reference_ref}
    assert sorted(assets) == sorted([_PNG, _PNG, _JPEG])
    assert downloads == ["https://example.com/reference.jpg"]


async def test_decide_rejects_non_image_files() -> None:
    model = _model(lambda request: httpx2.Response(500))
    document = ai.file_part(b"%PDF-1.7", media_type="application/pdf")

    with pytest.raises(ValueError, match="application/pdf"):
        await decisions.decide(
            model,
            {"contract": document},
            {"q": decisions.PredicateQuestion(instructions="Signed?")},
        )


async def test_decide_rejects_non_mapping_provider_options() -> None:
    model = _model(lambda request: httpx2.Response(500))

    with pytest.raises(TypeError, match="must be a mapping"):
        await decisions.decide(
            model,
            "state",
            {"q": decisions.PredicateQuestion(instructions="?")},
            params=decisions.DecideParams(
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
async def test_decide_maps_status_errors(
    status: int, error_type: type[ai.ProviderAPIError]
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={"detail": "nope"},
            headers={"x-typesafe-request-id": "req_123"},
        )

    with pytest.raises(error_type) as exc_info:
        await decisions.decide(
            _model(handler),
            "state",
            {"q": decisions.PredicateQuestion(instructions="?")},
        )

    exc = exc_info.value
    assert type(exc) is error_type
    assert exc.provider == "typesafe"
    assert exc.request_id == "req_123"
    assert exc.body == {"detail": "nope"}
    assert exc.http_context is not None
    assert exc.http_context.status_code == status


async def test_decide_maps_connection_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    with pytest.raises(ai.ProviderConnectionError) as exc_info:
        await decisions.decide(
            _model(handler),
            "state",
            {"q": decisions.PredicateQuestion(instructions="?")},
        )

    assert exc_info.value.is_retryable


async def test_decide_maps_invalid_responses() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model": "jev-1.13.0"})

    with pytest.raises(ai.errors.ProviderResponseError) as exc_info:
        await decisions.decide(
            _model(handler),
            "state",
            {"q": decisions.PredicateQuestion(instructions="?")},
        )

    assert not exc_info.value.is_retryable
