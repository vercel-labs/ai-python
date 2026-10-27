"""TypeSafe System One protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from ... import ops
from ...types import usage as usage_
from .. import base
from . import _sdk, errors

if TYPE_CHECKING:
    import typesafe_sdk

    from ...models.core import model as model_

    TypeSafeClient = typesafe_sdk.AsyncTypeSafeClient
else:
    TypeSafeClient = Any


class TypeSafeSystemOneProtocol(base.ProviderProtocol[TypeSafeClient]):
    """TypeSafe ``/v1/systemone`` evaluation protocol."""

    protocol_class_id: Literal["typesafe-system-one"] = "typesafe-system-one"

    async def evaluate(
        self,
        client: TypeSafeClient,
        model: model_.Model,
        state: ops.experimental.EvaluationInput,
        questions: Mapping[
            str,
            ops.experimental.ChoiceQuestion
            | ops.experimental.ScoreQuestion
            | ops.experimental.NoulQuestion,
        ],
        *,
        params: ops.experimental.EvaluationParams,
        provider: str,
    ) -> ops.items.Item[dict[str, Any]]:
        """Hit ``/v1/systemone`` and return raw evaluation answers."""
        typesafe = _sdk.import_sdk(provider=provider)

        # The native wire format matches the question models, except that an
        # absent Noul criteria is omitted rather than sent as null.
        wire_questions: dict[str, Any] = {}
        for question_id, question in questions.items():
            wire_question = question.model_dump(mode="json")
            if wire_question.get("criteria", ...) is None:
                del wire_question["criteria"]
            wire_questions[question_id] = wire_question

        # Provider options are extra top-level request body fields.
        options = params.provider_options.get(provider)
        if options is not None and not isinstance(options, Mapping):
            raise TypeError(f"provider_options[{provider!r}] must be a mapping")

        try:
            response = await client.system_one(
                state,
                wire_questions,
                model=model.id,
                extra_body=options,
            )
        except typesafe.TypeSafeError as exc:
            raise errors.map_error(
                exc, provider=provider, model_id=model.id
            ) from exc

        # JSON mode turns integer score level keys into the string keys used
        # by the shared answer models. Score legends only echo the question's
        # criteria, so they move to provider metadata.
        answers: dict[str, Any] = {}
        legends: dict[str, Any] = {}
        for question_id, answer in response.answers.items():
            data = answer.model_dump(mode="json")
            if "legend" in data:
                legends[question_id] = data.pop("legend")
            answers[question_id] = data

        typesafe_metadata: dict[str, Any] = {"model": response.model}
        if legends:
            typesafe_metadata["legends"] = legends
        return ops.items.Item(
            value=answers,
            usage=usage_.Usage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                raw=response.usage.model_dump(mode="json"),
            ),
            provider_metadata={provider: typesafe_metadata},
        )


__all__ = ["TypeSafeSystemOneProtocol"]
