"""Typed decisions about shared state via dedicated decision models.

::

    import ai
    import pydantic

    class Questions(pydantic.BaseModel):
        requests_refund: ai.ops.experimental.PredicateQuestion

    class Answers(pydantic.BaseModel):
        requests_refund: ai.ops.experimental.PredicateAnswer

    result = await ai.ops.experimental.decide(
        ai.get_model("typesafe-ai/jev"),
        {"message": "Please refund the duplicate charge."},
        Questions(
            requests_refund=ai.ops.experimental.PredicateQuestion(
                instructions="Is the customer requesting a refund?",
            ),
        ),
        output_type=Answers,
    )
    result.value.requests_refund

Images go anywhere in the state or question text as file parts::

    state = {"photo": ai.file_part(photo_bytes), "note": "Delivered today."}
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict, overload

import pydantic

from ... import experimental_telemetry as telemetry
from ...types import messages
from .. import items

if TYPE_CHECKING:
    from ...models.core import model as model_


_JSON_TYPES = frozenset((str, int, float, bool, list, dict))


def _decision_value_tag(value: Any) -> str:
    # Like pydantic.JsonValue: pick one union member by type, so bad input
    # gives one error at its path. Exact types take the fast path. Subclasses
    # validate as their base type; bool goes before int because bool
    # subclasses int. Other sequences and mappings validate as list and dict,
    # except binary data, which is never a list of ints.
    exact = type(value)
    if exact in _JSON_TYPES:
        return exact.__name__
    for type_ in (bool, int, float, str, messages.FilePart):
        if isinstance(value, type_):
            return type_.__name__
    if isinstance(value, Mapping):
        return "dict"
    if isinstance(value, Sequence) and not isinstance(
        value, bytes | bytearray | memoryview
    ):
        return "list"
    return "None" if value is None else "invalid"


_VALUE_DISCRIMINATOR = pydantic.Discriminator(
    _decision_value_tag,
    custom_error_type="invalid-decision-value",
    custom_error_message="input was not text, JSON, or a FilePart",
)
_INPUT_DISCRIMINATOR = pydantic.Discriminator(
    _decision_value_tag,
    custom_error_type="invalid-decision-input",
    custom_error_message=(
        "input was not text, a FilePart, a mapping, or a sequence"
    ),
)

if TYPE_CHECKING:
    # Read-only containers are covariant, so a ``list[str]`` or a
    # ``dict[str, int]`` is accepted as is.
    type _DecisionValue = (
        str
        | int
        | float
        | bool
        | messages.FilePart
        | Sequence[_DecisionValue]
        | Mapping[str, _DecisionValue]
        | None
    )
    type DecisionInput = (
        str
        | messages.FilePart
        | Mapping[str, _DecisionValue]
        | Sequence[_DecisionValue]
    )
    """Text, a file, or JSON data with files at any position.

    Files are :class:`~ai.types.messages.FilePart` objects, e.g. from
    :func:`ai.file_part`. A plain ``str`` is always text, never a URL. Any
    sequence becomes a list and any mapping a dict.
    """
else:
    # Validation stores every sequence as a list and every mapping as a dict.
    _AS_LIST = pydantic.BeforeValidator(list)
    _AS_DICT = pydantic.BeforeValidator(dict)

    type _DecisionValue = Annotated[
        Annotated[str, pydantic.Tag("str")]
        | Annotated[int, pydantic.Tag("int")]
        | Annotated[float, pydantic.Tag("float")]
        | Annotated[bool, pydantic.Tag("bool")]
        | Annotated[messages.FilePart, pydantic.Tag("FilePart")]
        | Annotated[list[_DecisionValue], _AS_LIST, pydantic.Tag("list")]
        | Annotated[dict[str, _DecisionValue], _AS_DICT, pydantic.Tag("dict")]
        | Annotated[None, pydantic.Tag("None")],
        _VALUE_DISCRIMINATOR,
    ]

    type DecisionInput = Annotated[
        Annotated[str, pydantic.Tag("str")]
        | Annotated[messages.FilePart, pydantic.Tag("FilePart")]
        | Annotated[dict[str, _DecisionValue], _AS_DICT, pydantic.Tag("dict")]
        | Annotated[list[_DecisionValue], _AS_LIST, pydantic.Tag("list")],
        _INPUT_DISCRIMINATOR,
    ]


class PredicateCriteria(TypedDict, total=False):
    """Optional descriptions of when a predicate answer is true or false."""

    true: DecisionInput | None
    false: DecisionInput | None


_QUESTION_CONFIG = pydantic.ConfigDict(
    allow_inf_nan=False,
    extra="forbid",
    frozen=True,
    strict=True,
)


class ChoiceQuestion(pydantic.BaseModel):
    """Choose one key from a nonempty map of options."""

    instructions: DecisionInput
    criteria: Annotated[
        dict[str, DecisionInput | None],
        pydantic.Field(min_length=1),
    ]
    type: Literal["choice"] = "choice"

    model_config = _QUESTION_CONFIG


class ScoreQuestion(pydantic.BaseModel):
    """Score state against at least two ordered rubric levels."""

    instructions: DecisionInput
    criteria: Annotated[
        list[DecisionInput | None],
        pydantic.Field(min_length=2),
    ]
    type: Literal["score"] = "score"

    model_config = _QUESTION_CONFIG


class PredicateQuestion(pydantic.BaseModel):
    """Estimate the probability that a statement about state is true."""

    instructions: DecisionInput
    criteria: PredicateCriteria | None = None
    type: Literal["predicate"] = "predicate"

    model_config = _QUESTION_CONFIG

    @pydantic.model_validator(mode="before")
    @classmethod
    def _accept_noul(cls, data: Any) -> Any:
        # Deprecated: accept the old "noul" type name.
        if isinstance(data, dict) and data.get("type") == "noul":
            warnings.warn(
                'type="noul" is deprecated, use type="predicate"',
                DeprecationWarning,
                stacklevel=2,
            )
            data = {**data, "type": "predicate"}
        return data


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
    confidence: Probability | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class ScoreAnswer(pydantic.BaseModel):
    """Fractional rubric score and its optional probability distribution."""

    type: Literal["score"] = "score"
    score: ScoreValue
    probabilities: dict[str, Probability] | None = None
    confidence: Probability | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class PredicateAnswer(pydantic.BaseModel):
    """Model-estimated probability that the answer is true."""

    type: Literal["predicate"] = "predicate"
    probability: Probability

    model_config = pydantic.ConfigDict(frozen=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _accept_noul(cls, data: Any) -> Any:
        # Deprecated: accept the old "noul" type name and answer field.
        if isinstance(data, dict) and (
            data.get("type") == "noul" or "noul" in data
        ):
            warnings.warn(
                'type="noul" and the noul field are deprecated, '
                'use type="predicate" and probability',
                DeprecationWarning,
                stacklevel=2,
            )
            data = {**data}
            if data.get("type") == "noul":
                data["type"] = "predicate"
            if "noul" in data:
                data["probability"] = data.pop("noul")
        return data

    @property
    def noul(self) -> float:
        """Deprecated alias for :attr:`probability`."""
        warnings.warn(
            "PredicateAnswer.noul is deprecated, use probability",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.probability


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecideParams:
    """Parameters for decisions."""

    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


type _Question = ChoiceQuestion | ScoreQuestion | PredicateQuestion
type _Answer = ChoiceAnswer | ScoreAnswer | PredicateAnswer

_QUESTION_TYPES = (ChoiceQuestion, ScoreQuestion, PredicateQuestion)
_ANSWER_TYPES: dict[type[_Question], type[pydantic.BaseModel]] = {
    ChoiceQuestion: ChoiceAnswer,
    ScoreQuestion: ScoreAnswer,
    PredicateQuestion: PredicateAnswer,
}
_INPUT_ADAPTER: pydantic.TypeAdapter[DecisionInput] = pydantic.TypeAdapter(
    DecisionInput,
    config=pydantic.ConfigDict(allow_inf_nan=False, strict=True),
)


@overload
async def decide[AnswerT: pydantic.BaseModel](
    model: model_.Model,
    state: DecisionInput,
    questions: pydantic.BaseModel,
    *,
    output_type: type[AnswerT],
    params: DecideParams | None = None,
) -> items.Item[AnswerT]: ...


@overload
async def decide(
    model: model_.Model,
    state: DecisionInput,
    questions: Mapping[str, pydantic.BaseModel],
    *,
    output_type: None = None,
    params: DecideParams | None = None,
) -> items.Item[dict[str, _Answer]]: ...


async def decide(
    model: model_.Model,
    state: DecisionInput,
    questions: pydantic.BaseModel | Mapping[str, pydantic.BaseModel],
    *,
    output_type: type[pydantic.BaseModel] | None = None,
    params: DecideParams | None = None,
) -> items.Item[Any]:
    """Decide questions about one shared state.

    Pass matching Pydantic question and output models for a statically typed
    value, or pass a question mapping to receive an answer mapping. Each choice,
    score, or predicate question is validated against its corresponding answer.

    ``state``, question ``instructions``, and ``criteria`` accept files
    (:class:`~ai.types.messages.FilePart`) alongside text and JSON, e.g.
    ``{"photo": ai.file_part(data), "note": "..."}``. Providers reject files
    they cannot send.

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
    params = params or DecideParams()
    data = telemetry.DecideSpanData(
        model=model.id,
        provider=model.provider.name,
        question_count=len(normalized),
    )
    async with telemetry.span(data) as sp:
        # Providers operate on one normalized question mapping and return raw
        # answer data.
        raw_item = await model.provider.decide(
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
                    answers[name] = PredicateAnswer.model_validate(raw_answer)
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
