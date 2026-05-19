"""AI Gateway-native provider-executed tools."""

from __future__ import annotations

from typing import Literal

import pydantic
from pydantic.alias_generators import to_camel

from ... import types

_CONFIG_MODEL = pydantic.ConfigDict(
    frozen=True,
    populate_by_name=True,
    alias_generator=to_camel,
)


class SourcePolicy(pydantic.BaseModel):
    """Source policy for controlling which domains to include/exclude."""

    model_config = _CONFIG_MODEL

    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    after_date: str | None = None


class Excerpts(pydantic.BaseModel):
    """Excerpt configuration for controlling result length."""

    model_config = _CONFIG_MODEL

    max_chars_per_result: int | None = None
    max_chars_total: int | None = None


class FetchPolicy(pydantic.BaseModel):
    """Fetch policy for controlling content freshness."""

    model_config = _CONFIG_MODEL

    max_age_seconds: int | None = None


class PerplexitySearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    max_results: int | None = None
    max_tokens_per_page: int | None = None
    max_tokens: int | None = None
    country: str | None = None
    search_domain_filter: list[str] | None = None
    search_language_filter: list[str] | None = None
    search_recency_filter: Literal["day", "week", "month", "year"] | None = None


class ParallelSearchArgs(pydantic.BaseModel):
    model_config = _CONFIG_MODEL

    mode: Literal["one-shot", "agentic"] | None = None
    max_results: int | None = None
    source_policy: SourcePolicy | None = None
    excerpts: Excerpts | None = None
    fetch_policy: FetchPolicy | None = None


def perplexity_search(
    *,
    max_results: int | None = None,
    max_tokens_per_page: int | None = None,
    max_tokens: int | None = None,
    country: str | None = None,
    search_domain_filter: list[str] | None = None,
    search_language_filter: list[str] | None = None,
    search_recency_filter: Literal["day", "week", "month", "year"]
    | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="perplexity_search",
        args=PerplexitySearchArgs(
            max_results=max_results,
            max_tokens_per_page=max_tokens_per_page,
            max_tokens=max_tokens,
            country=country,
            search_domain_filter=search_domain_filter,
            search_language_filter=search_language_filter,
            search_recency_filter=search_recency_filter,
        ),
    )


def parallel_search(
    *,
    mode: Literal["one-shot", "agentic"] | None = None,
    max_results: int | None = None,
    source_policy: SourcePolicy | dict[str, object] | None = None,
    excerpts: Excerpts | dict[str, object] | None = None,
    fetch_policy: FetchPolicy | dict[str, object] | None = None,
) -> types.tools.Tool:
    return types.tools.Tool(
        kind="provider",
        name="parallel_search",
        args=ParallelSearchArgs(
            mode=mode,
            max_results=max_results,
            source_policy=SourcePolicy.model_validate(source_policy)
            if isinstance(source_policy, dict)
            else source_policy,
            excerpts=Excerpts.model_validate(excerpts)
            if isinstance(excerpts, dict)
            else excerpts,
            fetch_policy=FetchPolicy.model_validate(fetch_policy)
            if isinstance(fetch_policy, dict)
            else fetch_policy,
        ),
    )


__all__ = [
    "Excerpts",
    "FetchPolicy",
    "ParallelSearchArgs",
    "PerplexitySearchArgs",
    "SourcePolicy",
    "parallel_search",
    "perplexity_search",
]
