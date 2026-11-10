"""Experimental model operations."""

import warnings
from typing import TYPE_CHECKING, Any

from .decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecideParams,
    DecisionInput,
    PredicateAnswer,
    PredicateCriteria,
    PredicateQuestion,
    ScoreAnswer,
    ScoreQuestion,
    decide,
)

# Deprecated names, kept as aliases of their replacements.
_DEPRECATED = {
    "evaluate": "decide",
    "EvaluationInput": "DecisionInput",
    "EvaluationParams": "DecideParams",
    "NoulAnswer": "PredicateAnswer",
    "NoulCriteria": "PredicateCriteria",
    "NoulQuestion": "PredicateQuestion",
}

if TYPE_CHECKING:
    evaluate = decide
    EvaluationInput = DecisionInput
    EvaluationParams = DecideParams
    NoulAnswer = PredicateAnswer
    NoulCriteria = PredicateCriteria
    NoulQuestion = PredicateQuestion
else:

    def __getattr__(name: str) -> Any:
        if name in _DEPRECATED:
            new = _DEPRECATED[name]
            warnings.warn(
                f"ai.ops.experimental.{name} is deprecated, "
                f"use ai.ops.experimental.{new}",
                DeprecationWarning,
                stacklevel=2,
            )
            return globals()[new]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecideParams",
    "DecisionInput",
    "PredicateAnswer",
    "PredicateCriteria",
    "PredicateQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "decide",
]
