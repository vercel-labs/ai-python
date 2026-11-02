"""TypeSafe System One protocol."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from ... import ops, types
from ...models import core
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
    """TypeSafe ``/v1/systemone`` decision protocol."""

    protocol_class_id: Literal["typesafe-system-one"] = "typesafe-system-one"

    async def decide(
        self,
        client: TypeSafeClient,
        model: model_.Model,
        state: ops.experimental.DecisionInput,
        questions: Mapping[
            str,
            ops.experimental.ChoiceQuestion
            | ops.experimental.ScoreQuestion
            | ops.experimental.PredicateQuestion,
        ],
        *,
        params: ops.experimental.DecideParams,
        provider: str,
    ) -> ops.items.Item[dict[str, Any]]:
        """Hit ``/v1/systemone`` and return raw decision answers."""
        typesafe = _sdk.import_sdk(provider=provider)

        # Files become SDK images wherever they are. The SDK moves them to
        # the request's assets and sends a reference in their place. One file
        # part gives one image, so a repeated file is sent once.
        images: dict[int, typesafe_sdk.Image] = {}

        async def to_wire(value: Any) -> Any:
            match value:
                case types.messages.FilePart(media_type=media_type, data=data):
                    if not media_type.startswith("image/"):
                        raise ValueError(
                            f"Unsupported media type for TypeSafe: {media_type}"
                        )
                    if id(value) not in images:
                        match data:
                            case str() if types.media.is_downloadable_url(data):
                                data, _ = await core.helpers.files.download(
                                    data
                                )
                            case str():
                                data = base64.b64decode(
                                    types.media.data_to_base64(data)
                                )
                        images[id(value)] = typesafe.Image(data)
                    return images[id(value)]
                case dict():
                    return {
                        key: await to_wire(item) for key, item in value.items()
                    }
                case list():
                    return [await to_wire(item) for item in value]
                case _:
                    return value

        wire_state = await to_wire(state)

        # The native wire format matches the question models, except that
        # TypeSafe calls the predicate primitive "noul" and an absent
        # predicate criteria is omitted rather than sent as null.
        wire_questions: dict[str, Any] = {}
        for question_id, question in questions.items():
            wire_question = await to_wire(dict(question))
            if wire_question["type"] == "predicate":
                wire_question["type"] = "noul"
            if wire_question.get("criteria", ...) is None:
                del wire_question["criteria"]
            wire_questions[question_id] = wire_question

        # Provider options are extra top-level request body fields.
        options = params.provider_options.get(provider)
        if options is not None and not isinstance(options, Mapping):
            raise TypeError(f"provider_options[{provider!r}] must be a mapping")

        try:
            response = await client.system_one(
                wire_state,
                wire_questions,
                model=model.id,
                # TypeSafe accepts images only with this opt-in header.
                extra_headers=(
                    {"TypeSafe-Danger-Zone-Images": "true"} if images else None
                ),
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
            if data.get("type") == "noul":
                data["type"] = "predicate"
                if "noul" in data:
                    data["probability"] = data.pop("noul")
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
