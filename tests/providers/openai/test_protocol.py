"""Tests for the OpenAI ``/v1/decisions`` protocol."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx2 as httpx
import pydantic
import pytest

import ai
from ai import ops

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_PNG_URL = "data:image/png;base64," + base64.b64encode(_PNG).decode()


def _model(handler: Any) -> ai.Model:
    provider = ai.get_provider(
        "openai",
        base_url="https://openai.test/v1",
        api_key="sk-test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return ai.Model(id="gpt-6-luna", provider=provider)


def _response(answers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "gpt-6-luna",
        "answers": answers,
        "usage": {
            "input_tokens": 42,
            "input_tokens_details": {
                "cached_tokens": 10,
                "cache_write_tokens": 2,
            },
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 42,
        },
    }


class Questions(pydantic.BaseModel):
    department: ops.experimental.ChoiceQuestion
    severity: ops.experimental.ScoreQuestion
    refund: ops.experimental.PredicateQuestion


class Answers(pydantic.BaseModel):
    department: ops.experimental.ChoiceAnswer
    severity: ops.experimental.ScoreAnswer
    refund: ops.experimental.PredicateAnswer


async def test_decide_sends_native_request_and_parses_answers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_response(
                [
                    {
                        "type": "choice",
                        "name": "department",
                        "choice": "billing",
                        "confidence": 0.8,
                        "probabilities": [
                            {"value": "billing", "probability": 0.9},
                            {"value": "support", "probability": 0.1},
                        ],
                    },
                    {
                        "type": "score",
                        "name": "severity",
                        "score": 1.1,
                        "confidence": 0.55,
                        "probabilities": [
                            {"value": 0, "label": "0", "probability": 0.1},
                            {"value": 1, "label": "1", "probability": 0.7},
                            {"value": 2, "label": "2", "probability": 0.2},
                        ],
                    },
                    {
                        "type": "predicate",
                        "name": "refund",
                        "probability": 0.95,
                    },
                ]
            ),
        )

    result = await ops.experimental.decide(
        _model(handler),
        "I was charged twice.",
        Questions(
            department=ops.experimental.ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments and refunds.", "support": None},
            ),
            severity=ops.experimental.ScoreQuestion(
                instructions="How severe is this?",
                criteria=["Cosmetic", "Workaround", None],
            ),
            refund=ops.experimental.PredicateQuestion(
                instructions="Does the customer want a refund?",
            ),
        ),
        output_type=Answers,
    )

    assert len(requests) == 1
    assert str(requests[0].url) == "https://openai.test/v1/decisions"
    assert json.loads(requests[0].content) == {
        "model": "gpt-6-luna",
        "input": "I was charged twice.",
        "questions": [
            {
                "type": "choice",
                "name": "department",
                "instructions": "Which team should handle this?",
                "choices": [
                    {
                        "value": "billing",
                        "description": "Payments and refunds.",
                    },
                    {"value": "support"},
                ],
            },
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe is this?",
                "levels": [
                    {"label": "0", "description": "Cosmetic"},
                    {"label": "1", "description": "Workaround"},
                    {"label": "2"},
                ],
            },
            {
                "type": "predicate",
                "name": "refund",
                "instructions": "Does the customer want a refund?",
            },
        ],
    }
    assert result.value == Answers(
        department=ops.experimental.ChoiceAnswer(
            choice="billing",
            confidence=0.8,
            probabilities={"billing": 0.9, "support": 0.1},
        ),
        severity=ops.experimental.ScoreAnswer(
            score=1.1,
            confidence=0.55,
            probabilities={"0": 0.1, "1": 0.7, "2": 0.2},
        ),
        refund=ops.experimental.PredicateAnswer(probability=0.95),
    )
    assert result.usage == ai.types.usage.Usage(
        input_tokens=42,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=10,
        cache_write_tokens=2,
        raw=_response([])["usage"],
    )
    assert result.provider_metadata == {"openai": {"model": "gpt-6-luna"}}


async def test_decide_sends_files_and_json_as_parts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_response(
                [{"type": "predicate", "name": "match", "probability": 0.99}]
            ),
        )

    await ops.experimental.decide(
        _model(handler),
        [
            "This is photo A.",
            ai.file_part(_PNG, media_type="image/png"),
            {"order": {"charges": [49, 49]}},
        ],
        {
            "match": ops.experimental.PredicateQuestion(
                instructions=[
                    "Is photo B a copy of photo A?",
                    ai.file_part(
                        "https://example.com/b.jpg", media_type="image/jpeg"
                    ),
                ],
            ),
        },
    )

    body = json.loads(requests[0].content)
    assert body["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "This is photo A."},
                {"type": "input_image", "image_url": _PNG_URL},
                {
                    "type": "input_json",
                    "data": {"order": {"charges": [49, 49]}},
                },
            ],
        }
    ]
    assert body["questions"][0]["instructions"] == [
        {"type": "input_text", "text": "Is photo B a copy of photo A?"},
        {"type": "input_image", "image_url": "https://example.com/b.jpg"},
    ]


@pytest.mark.parametrize(
    ("state", "content"),
    [
        ({"a": 1}, [{"type": "input_json", "data": {"a": 1}}]),
        (
            ai.file_part(_PNG, media_type="image/png"),
            [{"type": "input_image", "image_url": _PNG_URL}],
        ),
    ],
)
async def test_decide_sends_single_values_as_one_part(
    state: Any, content: list[dict[str, Any]]
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_response(
                [{"type": "predicate", "name": "ok", "probability": 0.5}]
            ),
        )

    await ops.experimental.decide(
        _model(handler),
        state,
        {"ok": ops.experimental.PredicateQuestion(instructions="OK?")},
    )

    assert json.loads(requests[0].content)["input"] == [
        {"role": "user", "content": content}
    ]


async def test_decide_returns_refusals() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(
                [
                    {"type": "refusal", "name": "unsafe"},
                    {"type": "predicate", "name": "safe", "probability": 0.1},
                ]
            ),
        )

    result = await ops.experimental.decide(
        _model(handler),
        "state",
        {
            "unsafe": ops.experimental.PredicateQuestion(instructions="A?"),
            "safe": ops.experimental.PredicateQuestion(instructions="B?"),
        },
    )

    assert result.value == {
        "unsafe": ops.experimental.RefusalAnswer(),
        "safe": ops.experimental.PredicateAnswer(probability=0.1),
    }


async def test_decide_sends_provider_options() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_response(
                [{"type": "predicate", "name": "ok", "probability": 0.5}]
            ),
        )

    await ops.experimental.decide(
        _model(handler),
        "state",
        {"ok": ops.experimental.PredicateQuestion(instructions="OK?")},
        params=ops.experimental.DecideParams(
            provider_options={
                "openai": {"prompt_cache_key": "tickets"},
                "gateway": {"ignored": True},
            }
        ),
    )

    body = json.loads(requests[0].content)
    assert body["prompt_cache_key"] == "tickets"
    assert "ignored" not in body


@pytest.mark.parametrize(
    ("state", "question", "match"),
    [
        (
            "state",
            ops.experimental.PredicateQuestion(
                instructions="OK?", criteria={"true": "Yes."}
            ),
            "predicate criteria",
        ),
        (
            "state",
            ops.experimental.ChoiceQuestion(
                instructions="Which?", criteria={"a": {"rule": 1}, "b": None}
            ),
            "text criteria",
        ),
        (
            ai.file_part(b"%PDF-1.4", media_type="application/pdf"),
            ops.experimental.PredicateQuestion(instructions="OK?"),
            "application/pdf",
        ),
        (
            {"photo": ai.file_part(_PNG, media_type="image/png")},
            ops.experimental.PredicateQuestion(instructions="OK?"),
            "files inside JSON",
        ),
        (
            ["text", 1],
            ops.experimental.PredicateQuestion(instructions="OK?"),
            "got int",
        ),
    ],
)
async def test_decide_rejects_unsupported_input(
    state: Any, question: Any, match: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request must not be sent")

    with pytest.raises(ValueError, match=match):
        await ops.experimental.decide(_model(handler), state, {"ok": question})


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ("bad", TypeError, "must be a mapping"),
        ({"model": "other"}, ValueError, "protocol-owned fields: model"),
    ],
)
async def test_decide_rejects_invalid_provider_options(
    options: Any, error: type[Exception], match: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request must not be sent")

    with pytest.raises(error, match=match):
        await ops.experimental.decide(
            _model(handler),
            "state",
            {"ok": ops.experimental.PredicateQuestion(instructions="OK?")},
            params=ops.experimental.DecideParams(
                provider_options={"openai": options}
            ),
        )


async def test_decide_maps_status_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": {"message": "slow down", "type": "rate"}}
        )

    with pytest.raises(ai.errors.ProviderRateLimitError):
        await ops.experimental.decide(
            _model(handler),
            "state",
            {"ok": ops.experimental.PredicateQuestion(instructions="OK?")},
        )


async def test_decide_rejects_unexpected_answer_names() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response(
                [{"type": "predicate", "name": "other", "probability": 0.5}]
            ),
        )

    with pytest.raises(ValueError, match="unexpected answer name: 'other'"):
        await ops.experimental.decide(
            _model(handler),
            "state",
            {"ok": ops.experimental.PredicateQuestion(instructions="OK?")},
        )
