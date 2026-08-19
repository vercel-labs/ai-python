"""Text embeddings via dedicated embedding models.

::

    import ai

    model = ai.get_model("openai/text-embedding-3-small")

    result = await ai.ops.embed(model, ["hello", "world"])
    result.value  # list[list[float]], same order as the input
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from .. import experimental_telemetry as telemetry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..models.core import model as model_
    from . import items


@dataclasses.dataclass(frozen=True, kw_only=True)
class EmbedParams:
    """Parameters for embedding generation."""

    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


async def embed(
    model: model_.Model,
    values: list[str],
    *,
    params: EmbedParams | None = None,
) -> items.Item[list[list[float]]]:
    """Embed text values with a dedicated embedding model.

    Returns an :class:`~ai.ops.Item` whose ``value`` is one embedding
    vector per input string, in input order. Embedding models report
    input token usage only.
    """
    data = telemetry.EmbedSpanData(
        model=model.id,
        provider=model.provider.name,
        input_count=len(values),
    )
    async with telemetry.span(data) as sp:
        item = await model.provider.embed(
            model, values, params=params or EmbedParams()
        )
        sp.data.usage = item.usage
        sp.data.output_count = len(item.value)
        if item.value:
            sp.data.dimensions = len(item.value[0])
        if item.warnings:
            sp.data.warnings = [w.model_dump() for w in item.warnings]
        return item
