"""Document reranking via dedicated reranking models.

::

    import ai

    model = ai.get_model("cohere/rerank-v3.5")
    documents = ["Paris is in France.", "Tokyo is in Japan."]

    result = await ai.ops.rerank(model, documents, "capital of Japan")
    [documents[r.index] for r in result.value]  # most relevant first
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pydantic

from .. import experimental_telemetry as telemetry
from . import items

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..models.core import model as model_


class RankedDocument(pydantic.BaseModel):
    """A document's position in the reranked order."""

    index: int
    """Index of the document in the original documents list."""
    score: float
    """Relevance score of the document against the query."""

    model_config = pydantic.ConfigDict(frozen=True)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RerankParams:
    """Parameters for reranking."""

    top_n: int | None = None
    """Return only the top N documents; all documents when ``None``."""
    provider_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    """Provider-specific options, keyed by provider name."""


async def rerank(
    model: model_.Model,
    documents: list[str] | list[dict[str, Any]],
    query: str,
    *,
    params: RerankParams | None = None,
) -> items.Item[list[RankedDocument]]:
    """Rerank documents against a query with a dedicated reranking model.

    ``documents`` is a list of texts or a list of JSON objects. Returns an
    :class:`~ai.ops.Item` whose ``value`` is one :class:`RankedDocument`
    per returned document, sorted by descending relevance score. Reranking
    models do not report token usage; cost information, when the provider
    sends it, is on ``.provider_metadata``.
    """
    if not documents:
        return items.Item(value=[])
    params = params or RerankParams()
    data = telemetry.RerankSpanData(
        model=model.id,
        provider=model.provider.name,
        input_count=len(documents),
        top_n=params.top_n,
    )
    async with telemetry.span(data) as sp:
        item = await model.provider.rerank(
            model, documents, query, params=params
        )
        sp.data.usage = item.usage
        sp.data.output_count = len(item.value)
        if item.warnings:
            sp.data.warnings = [w.model_dump() for w in item.warnings]
        return item
