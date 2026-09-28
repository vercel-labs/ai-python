"""Typed evaluation of shared state via dedicated evaluation models.

::

    import pydantic

    import ai

    class Answers(pydantic.BaseModel):
        requests_refund: ai.ops.experimental.BooleanAnswer

    class Questions(ai.ops.experimental.BaseQuestionModel[Answers]):
        requests_refund: ai.ops.experimental.BooleanQuestion

    result = await ai.ops.experimental.evaluate(
        ai.get_model("typesafe-ai/jev"),
        {"message": "Please refund the duplicate charge."},
        Questions(
            requests_refund=ai.ops.experimental.BooleanQuestion(
                instructions="Is the customer requesting a refund?",
            ),
        ),
    )
    result.value.requests_refund
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    TypedDict,
    cast,
    overload,
)

import pydantic

from ... import experimental_telemetry as telemetry
from .. import items

if TYPE_CHECKING:
    from ...models.core import model as model_


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


class BaseQuestionModel[AnswerT: pydantic.BaseModel](pydantic.BaseModel):
    """Base class for questions paired with an answer model.

    Fields must use ChoiceQuestion, ScoreQuestion, or BooleanQuestion, with
    matching names and corresponding answer types in the answer model.
    """

    __answers_type__: ClassVar[type[pydantic.BaseModel]]

    model_config = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_on_complete__(cls) -> None:  # noqa: PLW3201
        super().__pydantic_on_complete__()
        # Generic templates do not yet have a concrete answer model.
        if cls.__pydantic_generic_metadata__["parameters"]:
            return
        args = cls.__pydantic_generic_metadata__["args"]
        answers_type = (
            args[0] if args else getattr(cls, "__answers_type__", None)
        )
        if not isinstance(answers_type, type) or not issubclass(
            answers_type, pydantic.BaseModel
        ):
            raise TypeError(
                "questions must specialize BaseQuestionModel "
                "with a Pydantic model"
            )
        cls.__answers_type__ = answers_type

        # Pydantic creates BaseQuestionModel[Answers] before its subclass
        # defines the question fields. The answer model may be incomplete too.
        if cls.__pydantic_generic_metadata__["origin"] is BaseQuestionModel:
            return
        answers_type.model_rebuild()
        answer_fields = answers_type.model_fields
        for name, field in answer_fields.items():
            if not isinstance(field.annotation, type) or not issubclass(
                field.annotation, tuple(_ANSWER_TYPES.values())
            ):
                raise TypeError(
                    f"answer field {name!r} must be annotated as "
                    "ChoiceAnswer, ScoreAnswer, or BooleanAnswer"
                )

        question_fields = cls.model_fields
        for name, field in question_fields.items():
            if field.annotation not in _QUESTION_TYPES:
                raise TypeError(
                    f"question field {name!r} must be annotated as "
                    "ChoiceQuestion, ScoreQuestion, or BooleanQuestion"
                )
        if answer_fields.keys() != question_fields.keys():
            differences = []
            if missing_answers := question_fields.keys() - answer_fields.keys():
                differences.append(
                    f"missing from answer model {answers_type.__name__}: "
                    + ", ".join(repr(name) for name in sorted(missing_answers))
                )
            if (
                missing_questions := answer_fields.keys()
                - question_fields.keys()
            ):
                differences.append(
                    f"missing from question model {cls.__name__}: "
                    + ", ".join(
                        repr(name) for name in sorted(missing_questions)
                    )
                )
            raise TypeError(
                "answer model fields must match question fields; "
                + "; ".join(differences)
            )
        for name, field in question_fields.items():
            expected = _ANSWER_TYPES[cast("type[_Question]", field.annotation)]
            actual = answer_fields[name].annotation
            if not isinstance(actual, type) or not issubclass(actual, expected):
                raise TypeError(
                    f"answer field {name!r} must be annotated as "
                    f"{expected.__name__}"
                )


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
async def evaluate[AnswerT: pydantic.BaseModel](
    model: model_.Model,
    state: EvaluationInput | pydantic.BaseModel,
    questions: BaseQuestionModel[AnswerT],
    *,
    params: EvaluationParams | None = None,
) -> items.Item[AnswerT]: ...


@overload
async def evaluate(
    model: model_.Model,
    state: EvaluationInput | pydantic.BaseModel,
    questions: Mapping[str, pydantic.BaseModel],
    *,
    params: EvaluationParams | None = None,
) -> items.Item[dict[str, _Answer]]: ...


async def evaluate(
    model: model_.Model,
    state: EvaluationInput | pydantic.BaseModel,
    questions: BaseQuestionModel[Any] | Mapping[str, pydantic.BaseModel],
    *,
    params: EvaluationParams | None = None,
) -> items.Item[Any]:
    """Evaluate questions against one shared state.

    Pass a BaseQuestionModel parameterized with its Pydantic answer model for a
    statically typed value, or pass a question mapping to receive an answer
    mapping. Each question is validated against its corresponding answer.

    Experimental: not part of the stable API, may change or be removed.
    """
    # Normalize Pydantic state to JSON data before validating and dispatching.
    if isinstance(state, pydantic.BaseModel):
        state = state.model_dump(mode="json", by_alias=True)
    state = _INPUT_ADAPTER.validate_python(state)

    # Select one input mode and expose its question values for normalization.
    values: Iterable[tuple[str, Any]]
    answer_model: type[pydantic.BaseModel] | None = None
    if isinstance(questions, BaseQuestionModel):
        answer_model = questions.__answers_type__

        question_fields = type(questions).model_fields
        if not question_fields:
            raise ValueError("questions must not be empty")

        values = ((name, getattr(questions, name)) for name in question_fields)
    else:
        if not isinstance(questions, Mapping):
            raise TypeError("questions must be a BaseQuestionModel or mapping")
        if not questions:
            raise ValueError("questions must not be empty")
        values = questions.items()

    # Normalize either public input form to the mapping expected by providers.
    normalized: dict[str, _Question] = {}
    for name, question in values:
        if not isinstance(name, str):
            raise TypeError("question IDs must be strings")
        if not isinstance(question, _QUESTION_TYPES):
            raise TypeError(f"question field {name!r} must contain a question")

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

        if answer_model is not None:
            # Typed mode validates the complete response into the user's model.
            value: pydantic.BaseModel | dict[str, _Answer] = (
                answer_model.model_validate(raw_item.value)
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
