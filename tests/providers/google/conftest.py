"""Shared fakes for the Google adapter tests.

The adapter consumes ``google.genai.Client`` via
``client.aio.models.generate_content_stream(**kwargs)``. To exercise the
real adapter without hitting the network we build a tiny stand-in that
captures the kwargs and yields real SDK-typed response chunks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from google.genai import types as genai_types


def chunk(
    parts: list[genai_types.Part] | None = None,
    *,
    usage: dict[str, Any] | None = None,
    finish_reason: genai_types.FinishReason | None = None,
    block_reason: genai_types.BlockedReason | None = None,
) -> genai_types.GenerateContentResponse:
    """Build an SDK-typed streaming response chunk."""
    candidates = None
    if parts is not None or finish_reason is not None:
        candidates = [
            genai_types.Candidate(
                content=(
                    genai_types.Content(role="model", parts=parts)
                    if parts is not None
                    else None
                ),
                finish_reason=finish_reason,
            )
        ]
    return genai_types.GenerateContentResponse(
        candidates=candidates,
        prompt_feedback=(
            genai_types.GenerateContentResponsePromptFeedback(
                block_reason=block_reason
            )
            if block_reason is not None
            else None
        ),
        usage_metadata=(
            genai_types.GenerateContentResponseUsageMetadata(**usage)
            if usage is not None
            else None
        ),
    )


class FakeAsyncModels:
    def __init__(
        self,
        captured: dict[str, Any],
        chunks: list[genai_types.GenerateContentResponse],
    ) -> None:
        self._captured = captured
        self._chunks = chunks

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        self._captured.update(kwargs)

        async def _gen() -> Any:
            for item in self._chunks:
                yield item

        return _gen()


class FakeGoogleClient:
    """Stand-in for ``google.genai.Client``."""

    def __init__(
        self,
        captured: dict[str, Any] | None = None,
        chunks: list[genai_types.GenerateContentResponse] | None = None,
    ) -> None:
        self.captured = captured if captured is not None else {}
        self.aio = SimpleNamespace(
            models=FakeAsyncModels(self.captured, chunks or [])
        )
