from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pydantic
import pytest

import ai
from ai import models
from ai.types import events as events_
from ai.types import messages as messages_

from ...conftest import MOCK_MODEL, MockProvider, mock_llm, text_msg


def _test_provider_metadata(marker: str) -> dict[str, Any]:
    return {"marker": marker}


def _provider_metadata_marker(
    provider_metadata: dict[str, Any] | None,
) -> str:
    assert provider_metadata is not None
    marker = provider_metadata.get("marker")
    assert isinstance(marker, str)
    return marker


async def test_stream_aggregates_registered_adapter_events() -> None:
    mock = mock_llm([[text_msg("Hello world")]])

    deltas: list[str] = []
    async with models.stream(MOCK_MODEL, [ai.user_message("Hi")]) as stream:
        async for event in stream:
            if isinstance(event, events_.TextDelta):
                deltas.append(event.chunk)

        assert mock.call_count == 1
        assert stream.text == "Hello world"
        assert "".join(deltas) == "Hello world"


async def test_stream_tool_end_includes_aggregated_tool_call() -> None:
    async def _tool_stream(
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
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

    tool_end: events_.ToolEnd | None = None
    async with models.stream(MOCK_MODEL, [ai.user_message("Check weather")]) as stream:
        async for event in stream:
            if isinstance(event, events_.ToolEnd):
                tool_end = event

        assert tool_end is not None
        assert tool_end.tool_call.tool_call_id == "tc-1"
        assert tool_end.tool_call.tool_name == "weather"
        assert tool_end.tool_call.tool_args == '{"city":"SF"}'
        assert stream.tool_calls == [tool_end.tool_call]


async def test_stream_accumulates_provider_metadata_latest_wins() -> None:
    async def _metadata_stream() -> AsyncGenerator[events_.Event]:
        yield events_.StreamStart()
        yield events_.TextStart(
            block_id="text",
            provider_metadata=_test_provider_metadata("text-start"),
        )
        yield events_.TextDelta(block_id="text", chunk="hello")
        yield events_.TextDelta(
            block_id="text",
            chunk=" world",
            provider_metadata=_test_provider_metadata("text-delta"),
        )
        yield events_.TextEnd(
            block_id="text",
            provider_metadata=_test_provider_metadata("text-end"),
        )
        yield events_.ReasoningStart(
            block_id="reasoning",
            provider_metadata=_test_provider_metadata("reasoning-start"),
        )
        yield events_.ReasoningDelta(
            block_id="reasoning",
            chunk="thinking",
            provider_metadata=_test_provider_metadata("reasoning-delta"),
        )
        yield events_.ReasoningEnd(
            block_id="reasoning",
            provider_metadata=_test_provider_metadata("reasoning-end"),
        )
        yield events_.ToolStart(
            tool_call_id="tc-1",
            tool_name="weather",
            provider_metadata=_test_provider_metadata("tool-start"),
        )
        yield events_.ToolDelta(tool_call_id="tc-1", chunk='{"city"')
        yield events_.ToolDelta(
            tool_call_id="tc-1",
            chunk=':"SF"}',
            provider_metadata=_test_provider_metadata("tool-delta"),
        )
        yield events_.ToolEnd(
            tool_call_id="tc-1",
            tool_call=messages_.DUMMY_TOOL_CALL,
            provider_metadata=_test_provider_metadata("tool-end"),
        )
        yield events_.FileEvent(
            block_id="file",
            media_type="image/png",
            data="base64-data",
            provider_metadata=_test_provider_metadata("file"),
        )
        yield events_.StreamEnd(
            provider_metadata=_test_provider_metadata("message"),
        )

    stream = models.Stream(_metadata_stream())
    async for _ in stream:
        pass

    assert _provider_metadata_marker(stream.message.provider_metadata) == "message"

    text = stream.message.parts[0]
    assert isinstance(text, messages_.TextPart)
    assert text.text == "hello world"
    assert _provider_metadata_marker(text.provider_metadata) == "text-end"

    reasoning = stream.message.parts[1]
    assert isinstance(reasoning, messages_.ReasoningPart)
    assert reasoning.text == "thinking"
    assert _provider_metadata_marker(reasoning.provider_metadata) == "reasoning-end"

    tool_call = stream.message.parts[2]
    assert isinstance(tool_call, messages_.ToolCallPart)
    assert tool_call.tool_args == '{"city":"SF"}'
    assert _provider_metadata_marker(tool_call.provider_metadata) == "tool-end"

    file = stream.message.parts[3]
    assert isinstance(file, messages_.FilePart)
    assert file.data == "base64-data"
    assert _provider_metadata_marker(file.provider_metadata) == "file"


async def test_stream_forwards_output_type_and_request_params() -> None:
    received_output_types: list[type[pydantic.BaseModel] | None] = []
    received_params: list[Any] = []

    class Answer(pydantic.BaseModel):
        value: str

    async def _spy_stream(
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        received_output_types.append(output_type)
        received_params.append(kwargs.get("params"))
        yield events_.StreamStart()
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    params = {"raw": "ok"}
    async with models.stream(
        MOCK_MODEL,
        [ai.user_message("Hi")],
        output_type=Answer,
        params=params,
    ) as stream:
        async for _ in stream:
            pass

    assert received_output_types == [Answer]
    assert received_params == [params]


async def test_stream_accepts_context() -> None:
    """``stream(context=ctx)`` reads model/messages/tools off the context."""
    mock = mock_llm([[text_msg("ok")]])
    ctx = ai.Context(
        model=MOCK_MODEL,
        messages=[ai.user_message("Hi")],
        tools=[],
    )
    async with models.stream(context=ctx) as s:
        async for _ in s:
            pass
    assert mock.call_count == 1
    assert s.text == "ok"


async def test_stream_rejects_context_with_positional_args() -> None:
    """Passing both positional model/messages and ``context=`` is a TypeError."""
    ctx = ai.Context(
        model=MOCK_MODEL,
        messages=[ai.user_message("Hi")],
        tools=[],
    )
    with pytest.raises(TypeError, match="either model/messages/tools or context="):
        async with models.stream(  # type: ignore[call-overload]
            MOCK_MODEL, [ai.user_message("Hi")], context=ctx
        ):
            pass


async def test_stream_requires_model_messages_or_context() -> None:
    """Passing nothing is a TypeError."""
    with pytest.raises(TypeError, match="either model and messages or context="):
        async with models.stream():  # type: ignore[call-overload]
            pass


async def test_generate_dispatches_to_registered_adapter() -> None:
    provider = MockProvider(adapter="mock-generate")
    model = models.Model(
        id="generate-model",
        adapter="mock-generate",
        provider=provider,
    )
    sentinel = messages_.Message(
        role="assistant",
        parts=[messages_.FilePart(data=b"\x89PNG", media_type="image/png")],
    )
    called = False

    async def _generate(
        model: models.Model,
        messages: list[messages_.Message],
        params: Any = None,
    ) -> messages_.Message:
        nonlocal called
        called = True
        return sentinel

    models.register_generate("mock-generate", _generate)

    result = await models.generate(
        model,
        [ai.user_message("A cat")],
        models.ImageParams(n=1),
    )

    assert called
    assert result is sentinel


class _CheckProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__(adapter="mock-check")
        self.checked_model: models.Model | None = None

    async def probe(self, model: models.Model) -> None:
        self.checked_model = model


async def test_probe_delegates_to_model_provider() -> None:
    provider = _CheckProvider()
    model = models.Model("mock-model", provider=provider)

    await models.probe(model)

    assert provider.checked_model is model


async def test_stream_replays_marked_last_assistant_with_tool_calls() -> None:
    """If the last message is marked replay, no LLM call."""
    called = False

    async def _spy_stream(
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        nonlocal called
        called = True
        yield events_.StreamStart()
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    assistant_msg = messages_.Message(
        role="assistant",
        parts=[
            messages_.TextPart(id="t1", text="calling tools"),
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="weather",
                tool_args='{"city":"SF"}',
            ),
        ],
    )

    async with models.stream(
        MOCK_MODEL,
        [ai.user_message("Hi"), assistant_msg.model_copy(update={"replay": True})],
    ) as stream:
        events: list[events_.Event] = [event async for event in stream]

        assert called is False, "should not have hit the LLM"
        # Stream.message is seeded from the original turn — text and tool
        # calls are both preserved.
        assert stream.text == "calling tools"
        assert len(stream.tool_calls) == 1
        assert stream.tool_calls[0].tool_call_id == "tc-1"
        assert stream.tool_calls[0].tool_args == '{"city":"SF"}'
        # Replay only emits ToolEnd events, flagged for agent.run to drop.
        tool_ends = [e for e in events if isinstance(e, events_.ToolEnd)]
        assert len(tool_ends) == 1
        assert tool_ends[0].replay is True
        assert tool_ends[0].tool_call.tool_call_id == "tc-1"
        # No ToolStart/Delta/text events are re-emitted.
        assert not any(isinstance(e, events_.ToolStart) for e in events)
        assert not any(isinstance(e, events_.TextDelta) for e in events)


def test_tool_end_replay_flag_excluded_from_json() -> None:
    """The replay flag is internal — it must not appear in serialized output."""
    ev = events_.ToolEnd(
        tool_call_id="tc-1",
        tool_call=messages_.DUMMY_TOOL_CALL,
        replay=True,
    )
    dumped = ev.model_dump()
    assert "replay" not in dumped
    dumped_json = ev.model_dump(mode="json")
    assert "replay" not in dumped_json


async def test_stream_does_not_replay_when_assistant_is_unmarked() -> None:
    """Bare assistant text doesn't trigger replay — fall through to LLM."""
    called = False

    async def _spy_stream(
        model: models.Model,
        messages: list[messages_.Message],
        *,
        tools: Sequence[ai.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[events_.Event]:
        nonlocal called
        called = True
        yield events_.StreamStart()
        yield events_.StreamEnd()

    models.register_stream("mock", _spy_stream)

    assistant_text_only = messages_.Message(
        role="assistant",
        parts=[messages_.TextPart(text="just talking")],
    )

    async with models.stream(MOCK_MODEL, [assistant_text_only]) as stream:
        async for _ in stream:
            pass

    assert called is True
