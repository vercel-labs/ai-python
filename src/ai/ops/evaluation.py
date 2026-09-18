"""Typed evaluation of shared state via dedicated evaluation models.

::

    import ai
    import pydantic

    class Questions(pydantic.BaseModel):
        requests_refund: ai.ops.BooleanQuestion

    class Answers(pydantic.BaseModel):
        requests_refund: ai.ops.BooleanAnswer

    result = await ai.ops.experimental_evaluate(
        ai.get_model("typesafe-ai/jev"),
        {"message": "Please refund the duplicate charge."},
        Questions(
            requests_refund=ai.ops.BooleanQuestion(
                instructions="Is the customer requesting a refund?",
            ),
        ),
        output_type=Answers,
    )
    result.value.requests_refund
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

import pydantic

from .. import experimental_telemetry as telemetry
from . import items

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..models.core import model as model_


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


Probability = Annotated[
    pydantic.StrictFloat,
    pydantic.Field(ge=0, le=1, allow_inf_nan=False),
]
ScoreValue = Annotated[
    pydantic.StrictFloat,
    pydantic.Field(allow_inf_nan=False),
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvaluationParams:
    """Parameters for evaluation."""

    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


type _Question = ChoiceQuestion | ScoreQuestion | BooleanQuestion

_QUESTION_TYPES = (ChoiceQuestion, ScoreQuestion, BooleanQuestion)
_ANSWER_TYPES: dict[type[_Question], type[pydantic.BaseModel]] = {
    ChoiceQuestion: ChoiceAnswer,
    ScoreQuestion: ScoreAnswer,
    BooleanQuestion: BooleanAnswer,
}
_INPUT_ADAPTER: pydantic.TypeAdapter[EvaluationInput] = pydantic.TypeAdapter(
    EvaluationInput,
    config=pydantic.ConfigDict(allow_inf_nan=False, strict=True),
)


def _normalize_questions(
    questions: pydantic.BaseModel,
    output_type: type[pydantic.BaseModel],
) -> dict[str, _Question]:
    if not isinstance(questions, pydantic.BaseModel):
        raise TypeError("questions must be a Pydantic model")
    if not isinstance(output_type, type) or not issubclass(
        output_type, pydantic.BaseModel
    ):
        raise TypeError("output_type must be a Pydantic model class")

    question_fields = type(questions).model_fields
    if not question_fields:
        raise ValueError("questions must not be empty")
    if questions.model_extra:
        raise TypeError("questions must not contain extra fields")

    answer_fields = output_type.model_fields
    if answer_fields.keys() != question_fields.keys():
        raise TypeError("output_type fields must match question fields")

    normalized: dict[str, _Question] = {}
    for name in question_fields:
        question = getattr(questions, name)
        if not isinstance(question, _QUESTION_TYPES):
            raise TypeError(f"question field {name!r} must contain a question")
        expected = _ANSWER_TYPES[type(question)]
        actual = answer_fields[name].annotation
        if actual is not expected:
            raise TypeError(
                f"output field {name!r} must be annotated as "
                f"{expected.__name__}"
            )
        normalized[name] = question
    return normalized


async def experimental_evaluate[AnswerT: pydantic.BaseModel](
    model: model_.Model,
    state: EvaluationInput,
    questions: pydantic.BaseModel,
    *,
    output_type: type[AnswerT],
    params: EvaluationParams | None = None,
) -> items.Item[AnswerT]:
    """Evaluate typed questions against one shared state.

    ``questions`` and ``output_type`` must be Pydantic models with matching
    fields. Each choice, score, or boolean question must have the corresponding
    answer type. The returned item contains a validated ``output_type`` value.

    Experimental: not part of the stable API, may change or be removed.
    """
    state = _INPUT_ADAPTER.validate_python(state)
    normalized = _normalize_questions(questions, output_type)
    params = params or EvaluationParams()
    data = telemetry.EvaluateSpanData(
        model=model.id,
        provider=model.provider.name,
        question_count=len(normalized),
    )
    async with telemetry.span(data) as sp:
        raw_item = await model.provider.evaluate(
            model,
            state,
            normalized,
            params=params,
        )
        if raw_item.value.keys() != normalized.keys():
            raise ValueError("answer fields must match question fields")
        answer = output_type.model_validate(raw_item.value)
        item: items.Item[AnswerT] = items.Item(
            value=answer,
            usage=raw_item.usage,
            warnings=raw_item.warnings,
            metadata=raw_item.metadata,
            provider_metadata=raw_item.provider_metadata,
        )
        sp.data.usage = item.usage
        sp.data.answer_count = len(normalized)
        if item.warnings:
            sp.data.warnings = [
                warning.model_dump() for warning in item.warnings
            ]
        return item
