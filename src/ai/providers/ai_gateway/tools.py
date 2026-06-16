"""AI Gateway-native provider-executed tools."""

from __future__ import annotations

from typing import Any, Literal

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


def _dump[M: pydantic.BaseModel](
    model_type: type[M], value: M | dict[str, object] | None
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = model_type.model_validate(value)
    return value.model_dump(mode="json", exclude_none=True)


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
        tool_config=types.tools.ToolConfig(
            id="gateway.perplexity_search",
            args={
                k: v
                for k, v in {
                    "max_results": max_results,
                    "max_tokens_per_page": max_tokens_per_page,
                    "max_tokens": max_tokens,
                    "country": country,
                    "search_domain_filter": search_domain_filter,
                    "search_language_filter": search_language_filter,
                    "search_recency_filter": search_recency_filter,
                }.items()
                if v is not None
            },
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
        tool_config=types.tools.ToolConfig(
            id="gateway.parallel_search",
            args={
                k: v
                for k, v in {
                    "mode": mode,
                    "max_results": max_results,
                    "source_policy": _dump(SourcePolicy, source_policy),
                    "excerpts": _dump(Excerpts, excerpts),
                    "fetch_policy": _dump(FetchPolicy, fetch_policy),
                }.items()
                if v is not None
            },
        ),
    )


__all__ = [
    "Excerpts",
    "FetchPolicy",
    "SourcePolicy",
    "parallel_search",
    "perplexity_search",
]
