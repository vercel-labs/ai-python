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


class Answers(pydantic.BaseModel):
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

    class BooleanAnswers(pydantic.BaseModel):
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
    class EmptyAnswers(pydantic.BaseModel):
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


def test_question_model_stores_answer_type() -> None:
    assert Questions.__answers_type__ is Answers
    assert questions().__answers_type__ is Answers
    assert "__answers_type__" not in Questions.model_fields


def test_question_model_requires_concrete_answer_type() -> None:
    with pytest.raises(TypeError, match="specialize BaseQuestionModel"):

        class UnparameterizedQuestions(ops.experimental.BaseQuestionModel[Any]):
            refund: ops.experimental.BooleanQuestion


@pytest.mark.parametrize("explicit_rebuild", [True, False])
@pytest.mark.parametrize(
    ("question_type", "answer_type", "error"),
    [
        (
            ops.experimental.BooleanQuestion,
            ops.experimental.BooleanAnswer,
            None,
        ),
        (str, ops.experimental.BooleanAnswer, "question field 'refund'"),
        (ops.experimental.BooleanQuestion, str, "answer field 'refund'"),
        (
            ops.experimental.BooleanQuestion,
            ops.experimental.ScoreAnswer,
            "must be annotated as BooleanAnswer",
        ),
    ],
)
def test_question_model_resolves_forward_references(
    monkeypatch: pytest.MonkeyPatch,
    explicit_rebuild: bool,
    question_type: type[Any],
    answer_type: type[Any],
    error: str | None,
) -> None:
    answers_type = pydantic.create_model(
        "DeferredAnswers", refund=("DeferredAnswer", ...)
    )
    question_base = ops.experimental.BaseQuestionModel.__class_getitem__(
        answers_type
    )
    assert isinstance(question_base, type)
    questions_type = pydantic.create_model(
        "DeferredQuestions",
        __base__=question_base,
        refund=("DeferredQuestion", ...),
    )
    assert not answers_type.__pydantic_complete__
    assert not questions_type.__pydantic_complete__

    monkeypatch.setitem(globals(), "DeferredAnswer", answer_type)
    monkeypatch.setitem(globals(), "DeferredQuestion", question_type)
    if error is not None:
        with pytest.raises(TypeError, match=error):
            if explicit_rebuild:
                questions_type.model_rebuild()
            else:
                questions_type.model_validate({"refund": questions().refund})
        return

    if explicit_rebuild:
        questions_type.model_rebuild()
    instance = questions_type.model_validate({"refund": questions().refund})
    assert isinstance(instance, ops.experimental.BaseQuestionModel)
    assert instance.__answers_type__ is answers_type
    assert answers_type.__pydantic_complete__
    assert questions_type.__pydantic_complete__
    assert answers_type.model_fields["refund"].annotation is answer_type
    assert questions_type.model_fields["refund"].annotation is question_type


async def test_evaluate_inherited_question_model() -> None:
    class Intermediate(Questions):
        pass

    class InheritedQuestions(Intermediate):
        pass

    assert InheritedQuestions.__answers_type__ is Answers

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


@pytest.mark.parametrize(
    "annotation",
    [
        str,
        Any,
        ops.experimental.BooleanAnswer,
        list[ops.experimental.BooleanQuestion],
        ops.experimental.BooleanQuestion | None,
    ],
)
def test_question_model_requires_question_fields(annotation: Any) -> None:
    class BooleanAnswers(pydantic.BaseModel):
        answer: ops.experimental.BooleanAnswer

    with pytest.raises(TypeError, match="question field 'answer'"):
        pydantic.create_model(
            "InvalidQuestions",
            __base__=ops.experimental.BaseQuestionModel[BooleanAnswers],
            answer=(annotation, ...),
        )


@pytest.mark.parametrize(
    "annotation",
    [
        str,
        Any,
        ops.experimental.BooleanQuestion,
        list[ops.experimental.BooleanAnswer],
        ops.experimental.BooleanAnswer | None,
    ],
)
def test_question_model_requires_answer_fields(annotation: Any) -> None:
    invalid_answers = pydantic.create_model(
        "InvalidAnswers", answer=(annotation, ...)
    )
    question_base = ops.experimental.BaseQuestionModel.__class_getitem__(
        invalid_answers
    )
    assert isinstance(question_base, type)
    with pytest.raises(TypeError, match="answer field 'answer'"):
        pydantic.create_model(
            "InvalidQuestions",
            __base__=question_base,
            answer=(ops.experimental.BooleanQuestion, ...),
        )


async def test_evaluate_accepts_answer_subclasses() -> None:
    class ExplainedAnswer(ops.experimental.BooleanAnswer):
        reason: str

    class ExplainedAnswers(pydantic.BaseModel):
        refund: ExplainedAnswer

    class ExplainedQuestions(
        ops.experimental.BaseQuestionModel[ExplainedAnswers]
    ):
        refund: ops.experimental.BooleanQuestion

    model = models.Model(
        id="mock-evaluation-model",
        provider=StaticEvaluationProvider(
            answers={"refund": {"probability": 0.98, "reason": "Duplicate"}}
        ),
    )
    result = await ops.experimental.evaluate(
        model,
        "state",
        ExplainedQuestions(refund=questions().refund),
    )
    assert isinstance(result.value.refund, ExplainedAnswer)
    assert result.value.refund.reason == "Duplicate"


@pytest.mark.parametrize(
    ("answer_names", "question_names", "details"),
    [
        (
            ["shared"],
            ["shared", "severity", "refund"],
            "missing from answer model TestAnswers: 'refund', 'severity'",
        ),
        (
            ["shared", "severity", "refund"],
            ["shared"],
            "missing from question model TestQuestions: 'refund', 'severity'",
        ),
        (
            ["shared", "refund"],
            ["shared", "severity"],
            "missing from answer model TestAnswers: 'severity'; "
            "missing from question model TestQuestions: 'refund'",
        ),
    ],
)
def test_question_model_requires_matching_fields(
    answer_names: list[str], question_names: list[str], details: str
) -> None:
    answer_fields: dict[str, Any] = dict.fromkeys(
        answer_names, (ops.experimental.BooleanAnswer, ...)
    )
    question_fields: dict[str, Any] = dict.fromkeys(
        question_names, (ops.experimental.BooleanQuestion, ...)
    )
    answers_type = pydantic.create_model("TestAnswers", **answer_fields)
    question_base = ops.experimental.BaseQuestionModel.__class_getitem__(
        answers_type
    )
    assert isinstance(question_base, type)
    with pytest.raises(TypeError) as exc:
        pydantic.create_model(
            "TestQuestions",
            __base__=question_base,
            **question_fields,
        )
    assert str(exc.value) == (
        "answer model fields must match question fields; " + details
    )


def test_question_model_requires_matching_answer_types() -> None:
    class WrongAnswers(pydantic.BaseModel):
        department: ops.experimental.ChoiceAnswer
        severity: ops.experimental.BooleanAnswer
        refund: ops.experimental.BooleanAnswer

    with pytest.raises(TypeError, match="ScoreAnswer"):

        class MismatchedQuestions(
            ops.experimental.BaseQuestionModel[WrongAnswers]
        ):
            department: ops.experimental.ChoiceQuestion
            severity: ops.experimental.ScoreQuestion
            refund: ops.experimental.BooleanQuestion


def test_question_model_validates_inherited_fields() -> None:
    with pytest.raises(TypeError, match="fields must match"):

        class ExtraQuestions(Questions):
            extra: ops.experimental.BooleanQuestion

    with pytest.raises(TypeError, match="ChoiceAnswer"):

        class ChangedQuestions(Questions):
            severity: ops.experimental.ChoiceQuestion  # type: ignore[assignment]


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
