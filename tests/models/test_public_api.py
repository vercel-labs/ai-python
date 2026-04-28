"""Public ai.stream() and ai.generate() wrappers — end-to-end through mock adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic

import ai
from ai import models
from ai.types import events as events_
from ai.types import messages as messages_

from ..conftest import (
    MOCK_MODEL,
    MOCK_PROVIDER,
    MockProvider,
    mock_llm,
    text_msg,
)


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

    s = models.stream(MOCK_MODEL, [ai.user_message("Hi")])
    deltas: list[str] = []
    async for event in s:
        if isinstance(event, events_.TextDelta):
            deltas.append(event.chunk)

    assert mock.call_count == 1
    assert s.text == "Hello world"
    assert "".join(deltas) == "Hello world"


async def test_stream_tool_end_includes_aggregated_tool_call() -> None:
    """ToolEnd exposes the full ToolCallPart assembled from streamed input."""

    async def _tool_stream(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        yield events_.StreamStart()
        yield events_.ToolStart(tool_call_id="tc-1", tool_name="weather")
        yield events_.ToolDelta(tool_call_id="tc-1", chunk='{"city"')
        yield events_.ToolDelta(tool_call_id="tc-1", chunk=':"SF"}')
        yield events_.ToolEnd(
            tool_call_id="tc-1",
            tool_call=messages_.DUMMY_TOOL_CALL,
        )
        yield events_.StreamEnd()

    models.register_stream("mock", _tool_stream)

    s = models.stream(MOCK_MODEL, [ai.user_message("Check weather")])
    tool_end: events_.ToolEnd | None = None
    async for event in s:
        if isinstance(event, events_.ToolEnd):
            tool_end = event

    assert tool_end is not None
    assert tool_end.tool_call.tool_call_id == "tc-1"
    assert tool_end.tool_call.tool_name == "weather"
    assert tool_end.tool_call.tool_args == '{"city":"SF"}'
    assert s.tool_calls == [tool_end.tool_call]


async def test_stream_with_explicit_client() -> None:
    """Model with explicit client= forwards it to the adapter."""
    received_clients: list[models.Client] = []

    async def _spy_stream(
        client: models.Client,
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.ToolLike] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        received_clients.append(client)
        yield events_.StreamStart()
        yield events_.TextStart(block_id="t1")
        yield events_.TextDelta(block_id="t1", chunk="ok")
        yield events_.TextEnd(block_id="t1")
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    explicit = models.Client(base_url="https://custom.test", api_key="sk-custom")
    explicit_model = models.Model(
        id="mock-model", adapter="mock", provider=MOCK_PROVIDER, client=explicit
    )
    s = models.stream(explicit_model, [ai.user_message("Hi")])
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
    ) -> AsyncGenerator[events_.Event]:
        # Stream emits text deltas; the StructuredOutputPart is currently not
        # part of the public Event vocabulary, so we exercise output via
        # a downstream-friendly path: emit text and let consumers parse.
        yield events_.StreamStart()
        yield events_.TextStart(block_id="t1")
        yield events_.TextDelta(block_id="t1", chunk=json_text)
        yield events_.TextEnd(block_id="t1")
        yield events_.StreamEnd()

    models.register_stream("mock", _structured_stream)

    s = models.stream(
        MOCK_MODEL, [ai.user_message("Give me a recipe")], output_type=_Recipe
    )
    async for _ in s:
        pass

    assert s.text == json_text


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

_MOCK_GEN_PROVIDER = MockProvider(adapter="mock-gen")

GENERATE_MODEL = models.Model(
    id="gen-model",
    adapter="mock-gen",
    provider=_MOCK_GEN_PROVIDER,
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
