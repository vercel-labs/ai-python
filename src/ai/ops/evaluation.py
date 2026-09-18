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
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, overload

import pydantic

from .. import experimental_telemetry as telemetry
from . import items

if TYPE_CHECKING:
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
type _Answer = ChoiceAnswer | ScoreAnswer | BooleanAnswer

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


@overload
async def experimental_evaluate[AnswerT: pydantic.BaseModel](
    model: model_.Model,
    state: EvaluationInput,
    questions: pydantic.BaseModel,
    *,
    output_type: type[AnswerT],
    params: EvaluationParams | None = None,
) -> items.Item[AnswerT]: ...


@overload
async def experimental_evaluate(
    model: model_.Model,
    state: EvaluationInput,
    questions: Mapping[str, pydantic.BaseModel],
    *,
    output_type: None = None,
    params: EvaluationParams | None = None,
) -> items.Item[dict[str, _Answer]]: ...


async def experimental_evaluate(
    model: model_.Model,
    state: EvaluationInput,
    questions: pydantic.BaseModel | Mapping[str, pydantic.BaseModel],
    *,
    output_type: type[pydantic.BaseModel] | None = None,
    params: EvaluationParams | None = None,
) -> items.Item[Any]:
    """Evaluate questions against one shared state.

    Pass matching Pydantic question and output models for a statically typed
    value, or pass a question mapping to receive an answer mapping. Each choice,
    score, or boolean question is validated against its corresponding answer.

    Experimental: not part of the stable API, may change or be removed.
    """
    # Validate the shared state before dispatching it to a provider.
    state = _INPUT_ADAPTER.validate_python(state)

    # Select one input mode and expose its question values for normalization.
    values: Iterable[tuple[str, Any]]
    if isinstance(questions, pydantic.BaseModel):
        if output_type is None:
            raise TypeError("output_type is required for Pydantic questions")
        if not isinstance(output_type, type) or not issubclass(
            output_type, pydantic.BaseModel
        ):
            raise TypeError("output_type must be a Pydantic model class")

        question_fields = type(questions).model_fields
        if not question_fields:
            raise ValueError("questions must not be empty")
        if questions.model_extra:
            raise TypeError("questions must not contain extra fields")

        # Typed question and answer models must describe the same field names.
        answer_fields = output_type.model_fields
        if answer_fields.keys() != question_fields.keys():
            raise TypeError("output_type fields must match question fields")
        values = ((name, getattr(questions, name)) for name in question_fields)
    else:
        if output_type is not None:
            raise TypeError("output_type cannot be used with mapped questions")
        if not isinstance(questions, Mapping):
            raise TypeError("questions must be a Pydantic model or mapping")
        if not questions:
            raise ValueError("questions must not be empty")
        answer_fields = None
        values = questions.items()

    # Normalize either public input form to the mapping expected by providers.
    normalized: dict[str, _Question] = {}
    for name, question in values:
        if not isinstance(name, str):
            raise TypeError("question IDs must be strings")
        if not isinstance(question, _QUESTION_TYPES):
            raise TypeError(f"question field {name!r} must contain a question")

        # In typed mode, verify each question has its corresponding answer type.
        if answer_fields is not None:
            expected = _ANSWER_TYPES[type(question)]
            actual = answer_fields[name].annotation
            if actual is not expected:
                raise TypeError(
                    f"output field {name!r} must be annotated as "
                    f"{expected.__name__}"
                )
        normalized[name] = question

    # Start telemetry after local validation, immediately around provider work.
    params = params or EvaluationParams()
    data = telemetry.EvaluateSpanData(
        model=model.id,
        provider=model.provider.name,
        question_count=len(normalized),
    )
    async with telemetry.span(data) as sp:
        # Providers operate on one normalized question mapping and return raw
        # answer data.
        raw_item = await model.provider.evaluate(
            model,
            state,
            normalized,
            params=params,
        )

        # Every requested question must have exactly one returned answer.
        if raw_item.value.keys() != normalized.keys():
            raise ValueError("answer fields must match question fields")

        if output_type is not None:
            # Typed mode validates the complete response into the user's model.
            value: pydantic.BaseModel | dict[str, _Answer] = (
                output_type.model_validate(raw_item.value)
            )
        else:
            # Dynamic mode validates each value using its question's answer
            # type.
            answers: dict[str, _Answer] = {}
            for name, question in normalized.items():
                raw_answer = raw_item.value[name]
                if isinstance(question, ChoiceQuestion):
                    answers[name] = ChoiceAnswer.model_validate(raw_answer)
                elif isinstance(question, ScoreQuestion):
                    answers[name] = ScoreAnswer.model_validate(raw_answer)
                else:
                    answers[name] = BooleanAnswer.model_validate(raw_answer)
            value = answers

        # Rewrap the validated value without dropping operation metadata.
        item: items.Item[Any] = items.Item(
            value=value,
            usage=raw_item.usage,
            warnings=raw_item.warnings,
            metadata=raw_item.metadata,
            provider_metadata=raw_item.provider_metadata,
        )

        # Record normalized response details on the operation span.
        sp.data.usage = item.usage
        sp.data.answer_count = len(normalized)
        if item.warnings:
            sp.data.warnings = [
                warning.model_dump() for warning in item.warnings
            ]
        return item
