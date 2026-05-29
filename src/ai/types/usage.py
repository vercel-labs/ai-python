from __future__ import annotations

from typing import Any

import pydantic


class Usage(pydantic.BaseModel):
    """Normalized token usage from a single LLM call.

    Provides a provider-agnostic view of token consumption. Fields that a
    provider does not report are left as ``None`` (not zero) so callers
    can distinguish "not reported" from "zero tokens used".
    """

    model_config = pydantic.ConfigDict(frozen=True)

    # input_tokens includes both cached and uncached input tokens.
    input_tokens: int = 0
    output_tokens: int = 0

    # Optional breakdowns — not all providers report these.
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    # Pass-through of the raw provider usage payload so callers can access
    # provider-specific fields (e.g. OpenAI's accepted_prediction_tokens).
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """input_tokens + output_tokens (always consistent)."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Accumulate usage across multiple LLM calls."""

        def _add_optional(a: int | None, b: int | None) -> int | None:
            """Add two optional ints. Returns None only if both are None."""
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=_add_optional(
                self.reasoning_tokens, other.reasoning_tokens
            ),
            cache_read_tokens=_add_optional(
                self.cache_read_tokens, other.cache_read_tokens
            ),
            cache_write_tokens=_add_optional(
                self.cache_write_tokens, other.cache_write_tokens
            ),
            # Don't merge raw — it's per-call and provider-specific.
        )
