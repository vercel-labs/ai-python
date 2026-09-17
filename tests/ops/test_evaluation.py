"""Tests for ``ai.ops.evaluation`` dispatch and input validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import pydantic
import pytest

import ai
from ai import models, ops

from .. import conftest


class EvaluationProvider(models.Provider):
    provider_class_id: Literal["test-evaluation-provider"] = (
        "test-evaluation-provider"
    )
    name: str = "mock-evaluation"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def evaluate(
        self,
        model: models.Model,
        state: ops.EvaluationInput,
        questions: Mapping[str, ops.EvaluationQuestion],
        *,
        params: ops.EvaluationParams,
    ) -> ops.Item[ops.Evaluation]:
        assert state == {"message": "refund me"}
        assert set(questions) == {"department", "severity", "refund"}
        assert params.provider_options == {
            "gateway": {"zeroDataRetention": True}
        }
        return ops.Item(
            value=ops.Evaluation(
                answers={
                    "department": ops.ChoiceAnswer(
                        choice="billing",
                        probabilities={"billing": 0.9, "support": 0.1},
                    ),
                    "severity": ops.ScoreAnswer(
                        score=1.5,
                        probabilities={"0": 0.0, "1": 0.5, "2": 0.5},
                    ),
                    "refund": ops.BooleanAnswer(probability=0.98),
                }
            ),
            usage=ai.types.usage.Usage(input_tokens=12),
        )


class StaticEvaluationProvider(models.Provider):
    provider_class_id: Literal["test-static-evaluation-provider"] = (
        "test-static-evaluation-provider"
    )
    name: str = "mock-static-evaluation"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None
    evaluation: ops.Evaluation

    async def list_models(self) -> list[str]:
        return []

    async def evaluate(
        self,
        model: models.Model,
        state: ops.EvaluationInput,
        questions: Mapping[str, ops.EvaluationQuestion],
        *,
        params: ops.EvaluationParams,
    ) -> ops.Item[ops.Evaluation]:
        return ops.Item(value=self.evaluation)


def questions() -> dict[str, ops.EvaluationQuestion]:
    return {
        "department": ops.ChoiceQuestion(
            instructions="Which team should handle this?",
            criteria={
                "billing": {"includes": ["charges", "refunds"]},
                "support": None,
            },
        ),
        "severity": ops.ScoreQuestion(
            instructions={"task": "Rate severity"},
            criteria=["Cosmetic", "Workaround exists", "Blocking"],
        ),
        "refund": ops.BooleanQuestion(
            instructions="Is the customer requesting a refund?",
            criteria={"true": "Refund requested", "false": None},
        ),
    }


async def test_evaluate_dispatch_and_span(recorder: conftest.Recorder) -> None:
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    result = await ops.evaluate(
        model,
        {"message": "refund me"},
        questions(),
        params=ops.EvaluationParams(
            provider_options={"gateway": {"zeroDataRetention": True}}
        ),
    )

    department = result.value.answers["department"]
    assert isinstance(department, ops.ChoiceAnswer)
    assert department.choice == "billing"
    refund = result.value.answers["refund"]
    assert isinstance(refund, ops.BooleanAnswer)
    assert refund.probability == 0.98

    (span,) = recorder.ended
    assert isinstance(span.data, ai.experimental_telemetry.EvaluateSpanData)
    assert span.data.question_count == 3
    assert span.data.answer_count == 3
    assert span.data.usage == ai.types.usage.Usage(input_tokens=12)


async def test_evaluate_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="evaluation-test", provider=provider)

    with pytest.raises(NotImplementedError, match="evaluate"):
        await ops.evaluate(
            model,
            "state",
            {"answer": ops.BooleanQuestion(instructions="Is this valid?")},
        )


def test_choice_question_requires_criteria() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.ChoiceQuestion(instructions="Choose", criteria={})


def test_score_question_requires_two_levels() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.ScoreQuestion(instructions="Score", criteria=["only one"])


def test_boolean_question_rejects_unknown_criteria() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.BooleanQuestion(
            instructions="Decide",
            criteria=cast("ops.BooleanCriteria", {"maybe": "Maybe"}),
        )


def test_question_rejects_non_json_instructions() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.BooleanQuestion(
            instructions=cast("ops.EvaluationInput", object()),
        )


@pytest.mark.parametrize("state", [1, {"value": float("nan")}])
async def test_evaluate_validates_state(state: Any) -> None:
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(pydantic.ValidationError):
        await ops.evaluate(
            model,
            cast("ops.EvaluationInput", state),
            questions(),
        )


async def test_evaluate_requires_questions() -> None:
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(ValueError, match="questions must not be empty"):
        await ops.evaluate(model, "state", {})


async def test_evaluate_requires_question_models() -> None:
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(pydantic.ValidationError):
        await ops.evaluate(
            model,
            "state",
            cast(
                "Mapping[str, ops.EvaluationQuestion]",
                {"answer": {"type": "boolean", "instructions": "Decide"}},
            ),
        )


async def test_evaluate_trusts_provider_output() -> None:
    evaluation = ops.Evaluation(answers={})
    model = models.Model(
        id="mock-evaluation-model",
        provider=StaticEvaluationProvider(evaluation=evaluation),
    )

    result = await ops.evaluate(
        model,
        "state",
        {"answer": ops.BooleanQuestion(instructions="Is this valid?")},
    )

    assert result.value == evaluation


async def test_evaluate_rejects_cyclic_state() -> None:
    state: list[Any] = []
    state.append(state)
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(pydantic.ValidationError):
        await ops.evaluate(
            model,
            cast("ops.EvaluationInput", state),
            questions(),
        )
