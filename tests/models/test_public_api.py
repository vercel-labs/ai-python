"""Public ai.stream() and ai.generate() wrappers — end-to-end through mock adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import ai
from ai import models
from ai.types import messages as messages_

from ..conftest import MOCK_MODEL, mock_llm, text_msg


# Module-level model so StructuredOutputPart can resolve it by FQN.
class _Recipe(pydantic.BaseModel):
    name: str
    steps: list[str]


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


async def test_stream_basic() -> None:
    """ai.models.stream() yields deltas and exposes .text after iteration."""
    mock = mock_llm([[text_msg("Hello world")]])

    s = await models.stream(MOCK_MODEL, [ai.user_message("Hi")])
    deltas: list[str] = []
    async for msg in s:
        if msg.text_delta:
            deltas.append(msg.text_delta)

    assert mock.call_count == 1
    assert s.text == "Hello world"
    assert "".join(deltas) == "Hello world"


async def test_stream_with_explicit_client() -> None:
    """ai.models.stream(client=...) forwards the provided client to the adapter."""
    received_clients: list[models.Client] = []

    async def _spy_stream(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[messages_.Message]:
        received_clients.append(client)
        yield messages_.Message(
            id="m1",
            role="assistant",
            parts=[messages_.TextPart(text="ok", state="done")],
        )

    models.register_stream("mock", _spy_stream)

    explicit = models.Client(base_url="https://custom.test", api_key="sk-custom")
    s = await models.stream(MOCK_MODEL, [ai.user_message("Hi")], client=explicit)
    async for _ in s:
        pass

    assert len(received_clients) == 1
    assert received_clients[0] is explicit


async def test_stream_with_output_type() -> None:
    """ai.models.stream(output_type=...) exposes .output after iteration."""
    json_text = '{"name": "Pancakes", "steps": ["Mix", "Cook"]}'

    # Register an adapter that emits text + a StructuredOutputPart, mimicking
    # what a fully-wired adapter would do when output_type is requested.
    async def _structured_stream(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[messages_.Message]:
        text_part = messages_.TextPart(text=json_text, state="done")
        parts: list[messages_.Part] = [text_part]
        if output_type is not None:
            import json

            parts.append(
                messages_.StructuredOutputPart(
                    data=json.loads(json_text),
                    output_type_name=f"{output_type.__module__}.{output_type.__qualname__}",
                )
            )
        yield messages_.Message(id="m1", role="assistant", parts=parts)

    models.register_stream("mock", _structured_stream)

    s = await models.stream(
        MOCK_MODEL, [ai.user_message("Give me a recipe")], output_type=_Recipe
    )
    async for _ in s:
        pass

    assert s.output is not None
    assert isinstance(s.output, _Recipe)
    assert s.output.name == "Pancakes"
    assert s.output.steps == ["Mix", "Cook"]


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

GENERATE_MODEL = models.Model(
    id="gen-model",
    adapter="mock-gen",
    provider="mock",
    capabilities=("image",),
)


async def test_generate_dispatches_to_registered_adapter() -> None:
    """ai.generate() resolves the adapter from the registry and returns its Message."""
    sentinel = messages_.Message(
        role="assistant",
        parts=[messages_.FilePart(data=b"\x89PNG", media_type="image/png")],
    )
    called = False

    async def _mock_gen(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        params: Any = None,
    ) -> messages_.Message:
        nonlocal called
        called = True
        return sentinel

    models.register_generate("mock-gen", _mock_gen)

    result = await models.generate(
        GENERATE_MODEL, [ai.user_message("A cat")], models.ImageParams(n=1)
    )
    assert called
    assert result is sentinel
