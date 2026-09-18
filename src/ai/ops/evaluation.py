"""Typed evaluation of shared state via dedicated evaluation models.

::

    import ai

    model = ai.get_model("typesafe-ai/jev")
    result = await ai.ops.experimental_evaluate(
        model,
        {"message": "Please refund the duplicate charge."},
        {
            "department": ai.ops.ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "Charges and refunds", "other": None},
            ),
            "requests_refund": ai.ops.BooleanQuestion(
                instructions="Is the customer requesting a refund?",
            ),
        },
    )
    result.value.answers["department"]
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

import pydantic

from .. import experimental_telemetry as telemetry

if TYPE_CHECKING:
    from ..models.core import model as model_
    from . import items


type EvaluationInput = (
    str | dict[str, pydantic.JsonValue] | list[pydantic.JsonValue]
)


class BooleanCriteria(TypedDict, total=False):
    """Optional descriptions of when a boolean answer is true or false."""

    true: EvaluationInput | None
    false: EvaluationInput | None


_QUESTION_CONFIG = pydantic.ConfigDict(
    allow_inf_nan=False,
    extra="forbid",
    frozen=True,
    strict=True,
)


class ChoiceQuestion(pydantic.BaseModel):
    """Choose one key from a nonempty map of options."""

    instructions: EvaluationInput
    criteria: Annotated[
        dict[str, EvaluationInput | None],
        pydantic.Field(min_length=1),
    ]
    type: Literal["choice"] = "choice"

    model_config = _QUESTION_CONFIG


class ScoreQuestion(pydantic.BaseModel):
    """Score state against at least two ordered rubric levels."""

    instructions: EvaluationInput
    criteria: Annotated[
        list[EvaluationInput | None],
        pydantic.Field(min_length=2),
    ]
    type: Literal["score"] = "score"

    model_config = _QUESTION_CONFIG


class BooleanQuestion(pydantic.BaseModel):
    """Estimate the probability that a statement about state is true."""

    instructions: EvaluationInput
    criteria: BooleanCriteria | None = None
    type: Literal["boolean"] = "boolean"

    model_config = _QUESTION_CONFIG


EvaluationQuestion = Annotated[
    ChoiceQuestion | ScoreQuestion | BooleanQuestion,
    pydantic.Field(discriminator="type"),
]


Probability = Annotated[
    pydantic.StrictFloat,
    pydantic.Field(ge=0, le=1, allow_inf_nan=False),
]
ScoreValue = Annotated[
    pydantic.StrictFloat,
    pydantic.Field(allow_inf_nan=False),
]
RoundingPrecision = Annotated[
    pydantic.StrictInt,
    pydantic.Field(ge=0, le=15),
]


class ChoiceAnswer(pydantic.BaseModel):
    """Selected choice and its optional complete probability distribution."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability] | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class ScoreAnswer(pydantic.BaseModel):
    """Fractional rubric score and its optional probability distribution."""

    type: Literal["score"] = "score"
    score: ScoreValue
    probabilities: dict[str, Probability] | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class BooleanAnswer(pydantic.BaseModel):
    """Model-estimated probability that the answer is true."""

    type: Literal["boolean"] = "boolean"
    probability: Probability

    model_config = pydantic.ConfigDict(frozen=True)


EvaluationAnswer = Annotated[
    ChoiceAnswer | ScoreAnswer | BooleanAnswer,
    pydantic.Field(discriminator="type"),
]


class EvaluationRounding(pydantic.BaseModel):
    """Decimal precision applied by a provider to evaluation values."""

    probability_decimals: RoundingPrecision | None = pydantic.Field(
        default=None,
        validation_alias="probabilityDecimals",
    )
    score_decimals: RoundingPrecision | None = pydantic.Field(
        default=None,
        validation_alias="scoreDecimals",
    )

    model_config = pydantic.ConfigDict(frozen=True, populate_by_name=True)


class Evaluation(pydantic.BaseModel):
    """Answers returned for one shared state."""

    answers: dict[str, EvaluationAnswer]
    rounding: EvaluationRounding | None = None

    model_config = pydantic.ConfigDict(frozen=True)


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvaluationParams:
    """Parameters for evaluation."""

    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


type _EvaluationQuestionInstance = (
    pydantic.InstanceOf[ChoiceQuestion]
    | pydantic.InstanceOf[ScoreQuestion]
    | pydantic.InstanceOf[BooleanQuestion]
)
type _EvaluationQuestions = Mapping[str, _EvaluationQuestionInstance]

_INPUT_ADAPTER: pydantic.TypeAdapter[EvaluationInput] = pydantic.TypeAdapter(
    EvaluationInput,
    config=pydantic.ConfigDict(allow_inf_nan=False, strict=True),
)
_QUESTIONS_ADAPTER: pydantic.TypeAdapter[_EvaluationQuestions] = (
    pydantic.TypeAdapter(
        _EvaluationQuestions,
        config=pydantic.ConfigDict(strict=True),
    )
)


async def experimental_evaluate(
    model: model_.Model,
    state: EvaluationInput,
    questions: Mapping[str, EvaluationQuestion],
    *,
    params: EvaluationParams | None = None,
) -> items.Item[Evaluation]:
    """Evaluate typed questions against one shared state.

    The state, instructions, and criteria descriptions can be strings, JSON
    objects, or JSON arrays. Choice questions select a declared option, score
    questions return a fractional position on an ordered rubric, and boolean
    questions return the model-estimated probability of true.

    Experimental: not part of the stable API, may change or be removed.
    """
    state = _INPUT_ADAPTER.validate_python(state)
    questions = _QUESTIONS_ADAPTER.validate_python(questions)
    if not questions:
        raise ValueError("questions must not be empty")
    params = params or EvaluationParams()
    data = telemetry.EvaluateSpanData(
        model=model.id,
        provider=model.provider.name,
        question_count=len(questions),
    )
    async with telemetry.span(data) as sp:
        item = await model.provider.evaluate(
            model,
            state,
            questions,
            params=params,
        )
        sp.data.usage = item.usage
        sp.data.answer_count = len(item.value.answers)
        if item.warnings:
            sp.data.warnings = [
                warning.model_dump() for warning in item.warnings
            ]
        return item
