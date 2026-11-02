"""Tests for ``ai.ops.experimental.decide`` dispatch and input validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, assert_type, cast

import pydantic
import pytest

import ai
from ai import models, ops

from ... import conftest

type DecisionQuestion = (
    ops.experimental.ChoiceQuestion
    | ops.experimental.ScoreQuestion
    | ops.experimental.PredicateQuestion
)


class Questions(pydantic.BaseModel):
    department: ops.experimental.ChoiceQuestion
    severity: ops.experimental.ScoreQuestion
    refund: ops.experimental.PredicateQuestion


class Answers(pydantic.BaseModel):
    department: ops.experimental.ChoiceAnswer
    severity: ops.experimental.ScoreAnswer
    refund: ops.experimental.PredicateAnswer


class DecisionProvider(models.Provider):
    provider_class_id: Literal["test-decision-provider"] = (
        "test-decision-provider"
    )
    name: str = "mock-decision"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None

    async def list_models(self) -> list[str]:
        return []

    async def decide(
        self,
        model: models.Model,
        state: ops.experimental.DecisionInput,
        questions: Mapping[str, DecisionQuestion],
        *,
        params: ops.experimental.DecideParams,
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
                "refund": {"type": "predicate", "probability": 0.98},
            },
            usage=ai.types.usage.Usage(input_tokens=12),
            metadata={"rounding": {"probabilityDecimals": 2}},
            provider_metadata={"test": {"request_id": "req_1"}},
        )


class StaticDecisionProvider(models.Provider):
    provider_class_id: Literal["test-static-decision-provider"] = (
        "test-static-decision-provider"
    )
    name: str = "mock-static-decision"
    default_base_url: str = "http://mock.test"
    api_key_env: str | None = None
    answers: dict[str, Any]

    async def list_models(self) -> list[str]:
        return []

    async def decide(
        self,
        model: models.Model,
        state: ops.experimental.DecisionInput,
        questions: Mapping[str, DecisionQuestion],
        *,
        params: ops.experimental.DecideParams,
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
        refund=ops.experimental.PredicateQuestion(
            instructions="Is the customer requesting a refund?",
            criteria={"true": "Refund requested", "false": None},
        ),
    )


async def test_decide_dispatch_and_span(recorder: conftest.Recorder) -> None:
    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    result = await ops.experimental.decide(
        model,
        {"message": "refund me"},
        questions(),
        output_type=Answers,
        params=ops.experimental.DecideParams(
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
    assert isinstance(span.data, ai.experimental_telemetry.DecideSpanData)
    assert span.data.question_count == 3
    assert span.data.answer_count == 3
    assert span.data.usage == ai.types.usage.Usage(input_tokens=12)


async def test_decide_dynamic_questions() -> None:
    model = models.Model(id="mock-decision-model", provider=DecisionProvider())
    dynamic_questions = {
        "department": questions().department,
        "severity": questions().severity,
        "refund": questions().refund,
    }

    result = await ops.experimental.decide(
        model,
        {"message": "refund me"},
        dynamic_questions,
        params=ops.experimental.DecideParams(
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
                | ops.experimental.PredicateAnswer,
            ]
        ],
    )
    assert isinstance(result.value["department"], ops.experimental.ChoiceAnswer)
    assert result.value["department"].choice == "billing"
    assert isinstance(result.value["severity"], ops.experimental.ScoreAnswer)
    assert result.value["severity"].score == 1.5
    assert isinstance(result.value["refund"], ops.experimental.PredicateAnswer)
    assert result.value["refund"].probability == 0.98
    assert result.metadata == {"rounding": {"probabilityDecimals": 2}}


async def test_decide_raises_not_implemented() -> None:
    provider = ai.get_provider("openai", api_key="[redacted]")
    model = ai.Model(id="decision-test", provider=provider)

    class PredicateQuestions(pydantic.BaseModel):
        answer: ops.experimental.PredicateQuestion

    class PredicateAnswers(pydantic.BaseModel):
        answer: ops.experimental.PredicateAnswer

    with pytest.raises(NotImplementedError, match="decide"):
        await ops.experimental.decide(
            model,
            "state",
            PredicateQuestions(
                answer=ops.experimental.PredicateQuestion(
                    instructions="Is this valid?"
                )
            ),
            output_type=PredicateAnswers,
        )


def test_choice_question_requires_criteria() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.ChoiceQuestion(instructions="Choose", criteria={})


def test_score_question_requires_two_levels() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.ScoreQuestion(
            instructions="Score", criteria=["only one"]
        )


def test_predicate_question_rejects_unknown_criteria() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.PredicateQuestion(
            instructions="Decide",
            criteria=cast(
                "ops.experimental.PredicateCriteria", {"maybe": "Maybe"}
            ),
        )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("evaluate", "decide"),
        ("EvaluationInput", "DecisionInput"),
        ("EvaluationParams", "DecideParams"),
        ("NoulAnswer", "PredicateAnswer"),
        ("NoulCriteria", "PredicateCriteria"),
        ("NoulQuestion", "PredicateQuestion"),
    ],
)
def test_deprecated_aliases(old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=new):
        alias = getattr(ops.experimental, old)
    assert alias is getattr(ops.experimental, new)


def test_deprecated_noul_type() -> None:
    with pytest.warns(DeprecationWarning, match="predicate"):
        question = ops.experimental.PredicateQuestion.model_validate(
            {"type": "noul", "instructions": "Is this valid?"}
        )
    assert question.type == "predicate"

    with pytest.warns(DeprecationWarning, match="probability"):
        answer = ops.experimental.PredicateAnswer.model_validate(
            {"type": "noul", "noul": 0.5}
        )
    assert answer == ops.experimental.PredicateAnswer(probability=0.5)

    with pytest.warns(DeprecationWarning, match="probability"):
        assert answer.noul == 0.5


def test_question_rejects_non_json_instructions() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.experimental.PredicateQuestion(
            instructions=cast("ops.experimental.DecisionInput", object()),
        )


@pytest.mark.parametrize("state", [1, {"value": float("nan")}])
async def test_decide_validates_state(state: Any) -> None:
    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.decide(
            model,
            cast("ops.experimental.DecisionInput", state),
            questions(),
            output_type=Answers,
        )


async def test_decide_requires_question_values() -> None:
    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    with pytest.raises(TypeError, match="must contain a question"):
        await ops.experimental.decide(
            model,
            "state",
            cast(
                "Mapping[str, DecisionQuestion]",
                {"answer": "invalid"},
            ),
        )


async def test_decide_requires_questions() -> None:
    class EmptyQuestions(pydantic.BaseModel):
        pass

    class EmptyAnswers(pydantic.BaseModel):
        pass

    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    with pytest.raises(ValueError, match="questions must not be empty"):
        await ops.experimental.decide(
            model,
            "state",
            EmptyQuestions(),
            output_type=EmptyAnswers,
        )

    empty: dict[str, DecisionQuestion] = {}
    with pytest.raises(ValueError, match="questions must not be empty"):
        await ops.experimental.decide(model, "state", empty)


async def test_decide_rejects_mixed_modes() -> None:
    model = models.Model(id="mock-decision-model", provider=DecisionProvider())
    decide = cast("Any", ops.experimental.decide)

    with pytest.raises(TypeError, match="required for Pydantic"):
        await decide(model, "state", questions())

    with pytest.raises(TypeError, match="cannot be used with mapped"):
        await decide(
            model,
            "state",
            {"refund": questions().refund},
            output_type=Answers,
        )


async def test_decide_requires_question_fields() -> None:
    class InvalidQuestions(pydantic.BaseModel):
        answer: str

    class PredicateAnswers(pydantic.BaseModel):
        answer: ops.experimental.PredicateAnswer

    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    with pytest.raises(TypeError, match="must contain a question"):
        await ops.experimental.decide(
            model,
            "state",
            InvalidQuestions(answer="invalid"),
            output_type=PredicateAnswers,
        )


async def test_decide_requires_matching_output_fields() -> None:
    class MissingAnswers(pydantic.BaseModel):
        department: ops.experimental.ChoiceAnswer

    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    with pytest.raises(TypeError, match="fields must match"):
        await ops.experimental.decide(
            model,
            "state",
            questions(),
            output_type=MissingAnswers,
        )


async def test_decide_requires_matching_answer_types() -> None:
    class WrongAnswers(pydantic.BaseModel):
        department: ops.experimental.ChoiceAnswer
        severity: ops.experimental.PredicateAnswer
        refund: ops.experimental.PredicateAnswer

    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    with pytest.raises(TypeError, match="ScoreAnswer"):
        await ops.experimental.decide(
            model,
            "state",
            questions(),
            output_type=WrongAnswers,
        )


async def test_decide_validates_provider_output() -> None:
    model = models.Model(
        id="mock-decision-model",
        provider=StaticDecisionProvider(
            answers={
                "department": {"type": "choice", "choice": "billing"},
                "severity": {"type": "score", "score": 1.5},
                "refund": {"type": "predicate", "probability": 2.0},
            }
        ),
    )

    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.decide(
            model,
            "state",
            questions(),
            output_type=Answers,
        )

    dynamic_questions: dict[str, DecisionQuestion] = {
        "department": questions().department,
        "severity": questions().severity,
        "refund": questions().refund,
    }
    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.decide(model, "state", dynamic_questions)


async def test_decide_requires_matching_answer_fields() -> None:
    model = models.Model(
        id="mock-decision-model",
        provider=StaticDecisionProvider(answers={}),
    )

    with pytest.raises(ValueError, match="answer fields must match"):
        await ops.experimental.decide(
            model,
            "state",
            questions(),
            output_type=Answers,
        )


async def test_decide_rejects_cyclic_state() -> None:
    state: list[Any] = []
    state.append(state)
    model = models.Model(id="mock-decision-model", provider=DecisionProvider())

    with pytest.raises(pydantic.ValidationError):
        await ops.experimental.decide(
            model,
            cast("ops.experimental.DecisionInput", state),
            questions(),
            output_type=Answers,
        )
