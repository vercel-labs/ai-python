"""Tests for ``ai.ops.experimental.evaluate`` dispatch and input validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, assert_type, cast

import pydantic
import pytest

import ai
from ai import models, ops

from ... import conftest

type EvaluationQuestion = (
    ops.experimental.ChoiceQuestion
    | ops.experimental.ScoreQuestion
    | ops.experimental.BooleanQuestion
)


class Answers(ops.experimental.BaseAnswerModel):
    department: ops.experimental.ChoiceAnswer
    severity: ops.experimental.ScoreAnswer
    refund: ops.experimental.BooleanAnswer


class Questions(ops.experimental.BaseQuestionModel[Answers]):
    department: ops.experimental.ChoiceQuestion
    severity: ops.experimental.ScoreQuestion
    refund: ops.experimental.BooleanQuestion


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
        state: ops.experimental.EvaluationInput,
        questions: Mapping[str, EvaluationQuestion],
        *,
        params: ops.experimental.EvaluationParams,
    ) -> ops.Item[dict[str, Any]]:
        assert state == {"message": "refund me"}
        assert set(questions) == {"department", "severity", "refund"}
        assert params.provider_options == {
            "gateway": {"zeroDataRetention": True}
        }
        return ops.Item(
            value={
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "support": 0.1},
                },
                "severity": {
                    "type": "score",
                    "score": 1.5,
                    "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
                },
                "refund": {"type": "boolean", "probability": 0.98},
            },
            usage=ai.types.usage.Usage(input_tokens=12),
            metadata={"rounding": {"probabilityDecimals": 2}},
            provider_metadata={"test": {"request_id": "req_1"}},
        )


class StaticEvaluationProvider(models.Provider):
    provider_class_id: Literal["test-static-evaluation-provider"] = (
        "test-static-evaluation-provider"
    )
    name: str = "mock-static-evaluation"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None
    answers: dict[str, Any]

    async def list_models(self) -> list[str]:
        return []

    async def evaluate(
        self,
        model: models.Model,
        state: ops.experimental.EvaluationInput,
        questions: Mapping[str, EvaluationQuestion],
        *,
        params: ops.experimental.EvaluationParams,
    ) -> ops.Item[dict[str, Any]]:
        return ops.Item(value=self.answers)


def questions() -> Questions:
    return Questions(
        department=ops.experimental.ChoiceQuestion(
            instructions="Which team should handle this?",
            criteria={
                "billing": {"includes": ["charges", "refunds"]},
                "support": None,
            },
        ),
        severity=ops.experimental.ScoreQuestion(
            instructions={"task": "Rate severity"},
            criteria=["Cosmetic", "Workaround exists", "Blocking"],
        ),
        refund=ops.experimental.BooleanQuestion(
            instructions="Is the customer requesting a refund?",
            criteria={"true": "Refund requested", "false": None},
        ),
    )


async def test_evaluate_dispatch_and_span(recorder: conftest.Recorder) -> None:
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    state: ops.experimental.EvaluationInput = {"message": "refund me"}
    result = await ops.experimental.evaluate(
        model,
        state,
        questions(),
        params=ops.experimental.EvaluationParams(
            provider_options={"gateway": {"zeroDataRetention": True}}
        ),
    )

    assert_type(result, ops.Item[Answers])
    assert isinstance(result.value, Answers)
    assert result.value.department.choice == "billing"
    assert result.value.refund.probability == 0.98
    assert result.metadata == {"rounding": {"probabilityDecimals": 2}}
    assert result.provider_metadata == {"test": {"request_id": "req_1"}}

    (span,) = recorder.ended
    assert isinstance(span.data, ai.experimental_telemetry.EvaluateSpanData)
    assert span.data.question_count == 3
    assert span.data.answer_count == 3
    assert span.data.usage == ai.types.usage.Usage(input_tokens=12)


async def test_evaluate_dynamic_questions() -> None:
    class State(pydantic.BaseModel):
        message: str

    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )
    dynamic_questions = {
        "department": questions().department,
        "severity": questions().severity,
        "refund": questions().refund,
    }

    result = await ops.experimental.evaluate(
        model,
        State(message="refund me"),
        dynamic_questions,
        params=ops.experimental.EvaluationParams(
            provider_options={"gateway": {"zeroDataRetention": True}}
        ),
    )

    assert_type(
        result,
        ops.Item[
            dict[
                str,
                ops.experimental.ChoiceAnswer
                | ops.experimental.ScoreAnswer
                | ops.experimental.BooleanAnswer,
            ]
        ],
    )
    assert isinstance(result.value["department"], ops.experimental.ChoiceAnswer)
    assert result.value["department"].choice == "billing"
    assert isinstance(result.value["severity"], ops.experimental.ScoreAnswer)
    assert result.value["severity"].score == 1.5
    assert isinstance(result.value["refund"], ops.experimental.BooleanAnswer)
    assert result.value["refund"].probability == 0.98
    assert result.metadata == {"rounding": {"probabilityDecimals": 2}}


async def test_evaluate_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="evaluation-test", provider=provider)

    class BooleanAnswers(ops.experimental.BaseAnswerModel):
        answer: ops.experimental.BooleanAnswer

    class BooleanQuestions(ops.experimental.BaseQuestionModel[BooleanAnswers]):
        answer: ops.experimental.BooleanQuestion

    with pytest.raises(NotImplementedError, match="evaluate"):
        await ops.experimental.evaluate(
            model,
            "state",
            BooleanQuestions(
                answer=ops.experimental.BooleanQuestion(
                    instructions="Is this valid?"
                )
            ),
        )


def test_choice_question_requires_criteria() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.ChoiceQuestion(instructions="Choose", criteria={})


def test_score_question_requires_two_levels() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.ScoreQuestion(
            instructions="Score", criteria=["only one"]
        )


def test_boolean_question_rejects_unknown_criteria() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.BooleanQuestion(
            instructions="Decide",
            criteria=cast(
                "ops.experimental.BooleanCriteria", {"maybe": "Maybe"}
            ),
        )


def test_question_rejects_non_json_instructions() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.BooleanQuestion(
            instructions=cast("ops.experimental.EvaluationInput", object()),
        )


@pytest.mark.parametrize("state", [1, {"value": float("nan")}])
async def test_evaluate_validates_state(state: Any) -> None:
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.evaluate(
            model,
            cast("ops.experimental.EvaluationInput", state),
            questions(),
        )


async def test_evaluate_requires_question_values() -> None:
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(TypeError, match="must contain a question"):
        await ops.experimental.evaluate(
            model,
            "state",
            cast(
                "Mapping[str, EvaluationQuestion]",
                {"answer": "invalid"},
            ),
        )


async def test_evaluate_requires_questions() -> None:
    class EmptyAnswers(ops.experimental.BaseAnswerModel):
        pass

    class EmptyQuestions(ops.experimental.BaseQuestionModel[EmptyAnswers]):
        pass

    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(ValueError, match="questions must not be empty"):
        await ops.experimental.evaluate(
            model,
            "state",
            EmptyQuestions(),
        )

    empty: dict[str, EvaluationQuestion] = {}
    with pytest.raises(ValueError, match="questions must not be empty"):
        await ops.experimental.evaluate(model, "state", empty)


async def test_evaluate_requires_base_question_model() -> None:
    class PlainQuestions(pydantic.BaseModel):
        refund: ops.experimental.BooleanQuestion

    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(TypeError, match="BaseQuestionModel or mapping"):
        await ops.experimental.evaluate(
            model,
            "state",
            cast("Any", PlainQuestions(refund=questions().refund)),
        )


async def test_evaluate_requires_concrete_answer_type() -> None:
    class UnparameterizedQuestions(ops.experimental.BaseQuestionModel[Any]):
        refund: ops.experimental.BooleanQuestion

    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(TypeError, match="specialize BaseQuestionModel"):
        await ops.experimental.evaluate(
            model,
            "state",
            UnparameterizedQuestions(refund=questions().refund),
        )


async def test_evaluate_inherited_question_model() -> None:
    class Intermediate(ops.experimental.BaseQuestionModel[Answers]):
        pass

    class InheritedQuestions(Intermediate):
        department: ops.experimental.ChoiceQuestion
        severity: ops.experimental.ScoreQuestion
        refund: ops.experimental.BooleanQuestion

    model = models.Model(
        id="mock-evaluation-model",
        provider=StaticEvaluationProvider(
            answers={
                "department": {"choice": "billing"},
                "severity": {"score": 1.5},
                "refund": {"probability": 0.98},
            }
        ),
    )
    result = await ops.experimental.evaluate(
        model,
        "state",
        InheritedQuestions.model_validate(questions().model_dump()),
    )
    assert_type(result, ops.Item[Answers])
    assert isinstance(result.value, Answers)
    assert result.value.refund.probability == 0.98


async def test_evaluate_requires_question_fields() -> None:
    class BooleanAnswers(ops.experimental.BaseAnswerModel):
        answer: ops.experimental.BooleanAnswer

    class InvalidQuestions(ops.experimental.BaseQuestionModel[BooleanAnswers]):
        answer: str

    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(TypeError, match="must contain a question"):
        await ops.experimental.evaluate(
            model,
            "state",
            InvalidQuestions(answer="invalid"),
        )


async def test_evaluate_requires_matching_output_fields() -> None:
    class MissingAnswers(ops.experimental.BaseAnswerModel):
        department: ops.experimental.ChoiceAnswer

    class MismatchedQuestions(
        ops.experimental.BaseQuestionModel[MissingAnswers]
    ):
        department: ops.experimental.ChoiceQuestion
        severity: ops.experimental.ScoreQuestion
        refund: ops.experimental.BooleanQuestion

    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(TypeError, match="fields must match"):
        await ops.experimental.evaluate(
            model,
            "state",
            MismatchedQuestions.model_validate(questions().model_dump()),
        )


async def test_evaluate_requires_matching_answer_types() -> None:
    class WrongAnswers(ops.experimental.BaseAnswerModel):
        department: ops.experimental.ChoiceAnswer
        severity: ops.experimental.BooleanAnswer
        refund: ops.experimental.BooleanAnswer

    class MismatchedQuestions(ops.experimental.BaseQuestionModel[WrongAnswers]):
        department: ops.experimental.ChoiceQuestion
        severity: ops.experimental.ScoreQuestion
        refund: ops.experimental.BooleanQuestion

    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(TypeError, match="ScoreAnswer"):
        await ops.experimental.evaluate(
            model,
            "state",
            MismatchedQuestions.model_validate(questions().model_dump()),
        )


async def test_evaluate_validates_provider_output() -> None:
    model = models.Model(
        id="mock-evaluation-model",
        provider=StaticEvaluationProvider(
            answers={
                "department": {"type": "choice", "choice": "billing"},
                "severity": {"type": "score", "score": 1.5},
                "refund": {"type": "boolean", "probability": 2.0},
            }
        ),
    )

    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.evaluate(
            model,
            "state",
            questions(),
        )

    dynamic_questions: dict[str, EvaluationQuestion] = {
        "department": questions().department,
        "severity": questions().severity,
        "refund": questions().refund,
    }
    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.evaluate(model, "state", dynamic_questions)


async def test_evaluate_requires_matching_answer_fields() -> None:
    model = models.Model(
        id="mock-evaluation-model",
        provider=StaticEvaluationProvider(answers={}),
    )

    with pytest.raises(ValueError, match="answer fields must match"):
        await ops.experimental.evaluate(
            model,
            "state",
            questions(),
        )


async def test_evaluate_rejects_cyclic_state() -> None:
    state: list[Any] = []
    state.append(state)
    model = models.Model(
        id="mock-evaluation-model", provider=EvaluationProvider()
    )

    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.evaluate(
            model,
            cast("ops.experimental.EvaluationInput", state),
            questions(),
        )
