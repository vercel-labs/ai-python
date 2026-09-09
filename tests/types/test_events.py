"""Tests for ``ai.types.events``."""

from __future__ import annotations

import pydantic
import pytest

from ai import models
from ai.types import events, messages


def test_omitted_message_only_serializes_id() -> None:
    adapter: pydantic.TypeAdapter[events.AgentEvent] = pydantic.TypeAdapter(
        events.OmitEventMessages[events.AgentEvent]
    )
    message = messages.Message(
        id="message-1",
        role="assistant",
        parts=[messages.TextPart(text="hello")],
    )
    event = events.TextDelta(message=message, chunk="hello")

    result = adapter.dump_python(event)

    assert result["message"] == {"id": "message-1"}


def test_omit_event_messages_annotation_omits_message() -> None:
    adapter: pydantic.TypeAdapter[events.AgentEvent] = pydantic.TypeAdapter(
        events.OmitEventMessages[events.AgentEvent]
    )
    event = events.TextDelta(chunk="hello")

    assert adapter.dump_python(event)["message"] == {"id": "<unset>"}
    assert b'"message":{"id":"<unset>"}' in adapter.dump_json(event)


async def test_message_hydrator_round_trips_replayed_message() -> None:
    original = messages.Message(
        id="message-1",
        role="assistant",
        parts=[
            messages.ReasoningPart(
                id="reasoning-1",
                text="thinking",
                provider_metadata={"provider": {"signature": "sig"}},
            ),
            messages.TextPart(id="text-1", text="hello world"),
            messages.ToolCallPart(
                id="tool-1",
                tool_call_id="tool-1",
                tool_name="weather",
                tool_args='{"city":"SF"}',
            ),
            messages.BuiltinToolCallPart(
                id="builtin-1",
                tool_call_id="builtin-1",
                tool_name="web_search",
                tool_args='{"query":"weather"}',
            ),
            messages.BuiltinToolReturnPart(
                id="builtin-result-1",
                tool_call_id="builtin-1",
                tool_name="web_search",
                result={"temperature": 65},
            ),
            messages.FilePart(
                id="file-1",
                data="https://example.com/chart.png",
                media_type="image/png",
                filename="chart.png",
            ),
        ],
        provider_metadata={"provider": {"response_id": "response-1"}},
    )
    originals = [
        original,
        messages.Message(
            id="message-2",
            role="assistant",
            parts=[messages.TextPart(id="text-2", text="goodbye")],
        ),
    ]
    compact: pydantic.TypeAdapter[events.Event] = pydantic.TypeAdapter(
        events.OmitEventMessages[events.Event]
    )
    regular: pydantic.TypeAdapter[events.Event] = pydantic.TypeAdapter(
        events.Event
    )
    hydrator = events.MessageHydrator()

    for message in originals:
        async with models.Stream.replay_message(message) as stream:
            async for event in stream:
                dumped = compact.dump_json(event)
                validated = regular.validate_json(dumped)
                assert validated.message.id == message.id
                assert validated.message.parts == []
                hydrator.feed(validated)

    assert hydrator.message == originals[-1]
    assert hydrator.messages == originals
    assert hydrator.messages_by_id == {
        message.id: message for message in originals
    }


def test_message_hydrator_reconstructs_omitted_messages() -> None:
    compact: pydantic.TypeAdapter[events.Event] = pydantic.TypeAdapter(
        events.OmitEventMessages[events.Event]
    )
    regular: pydantic.TypeAdapter[events.Event] = pydantic.TypeAdapter(
        events.Event
    )
    message = messages.Message(id="message-1", role="assistant", parts=[])
    source: list[events.Event] = [
        events.StreamStart(message=message),
        events.TextStart(message=message, block_id="text-1"),
        events.TextDelta(message=message, block_id="text-1", chunk="hello"),
        events.TextDelta(message=message, block_id="text-1", chunk=" world"),
        events.TextEnd(message=message, block_id="text-1"),
        events.StreamEnd(
            message=message,
            provider_metadata={"provider": {"id": "1"}},
        ),
    ]
    hydrator = events.MessageHydrator()

    hydrated = [
        hydrator.feed(regular.validate_json(compact.dump_json(event)))
        for event in source
    ]

    assert hydrator.message.id == "message-1"
    assert hydrator.message.text == "hello world"
    assert hydrator.message.provider_metadata == {"provider": {"id": "1"}}
    assert all(
        isinstance(event, events.ModelEvent)
        and event.message is hydrator.message
        for event in hydrated
    )
    assert hydrator.ended


def test_message_hydrator_selects_message_without_stream_start() -> None:
    message = messages.Message(id="message", role="assistant", parts=[])
    hydrator = events.MessageHydrator()

    hydrator.feed(events.TextStart(message=message, block_id="text"))
    hydrated = hydrator.feed(
        events.TextDelta(message=message, block_id="text", chunk="hello")
    )

    assert hydrated.message is hydrator.message
    assert hydrator.message.id == "message"
    assert hydrator.message.text == "hello"


def test_message_hydrator_rejects_mismatched_seed_id() -> None:
    hydrator = events.MessageHydrator(
        messages.Message(id="seed", role="assistant", parts=[])
    )

    with pytest.raises(ValueError, match=r"seed.*'seed'.*'stream'"):
        hydrator.feed(
            events.StreamStart(
                message=messages.Message(
                    id="stream", role="assistant", parts=[]
                )
            )
        )


def test_message_hydrator_accepts_matching_seed_id() -> None:
    seed = messages.Message(id="message", role="assistant", parts=[])
    hydrator = events.MessageHydrator(seed)

    hydrated = hydrator.feed(events.StreamStart(message=seed))

    assert isinstance(hydrated, events.StreamStart)
    assert hydrated.message is seed


def test_message_hydrator_tracks_multiple_messages() -> None:
    hydrator = events.MessageHydrator()
    tool_message = messages.Message(id="tool-message", role="tool", parts=[])
    hook_message = messages.Message(
        id="hook-message", role="internal", parts=[]
    )
    assistant_1 = messages.Message(id="assistant-1", role="assistant", parts=[])
    assistant_2 = messages.Message(id="assistant-2", role="assistant", parts=[])
    source: list[events.AgentEvent] = [
        events.StreamStart(message=assistant_1),
        events.TextStart(message=assistant_1, block_id="text-1"),
        events.TextDelta(message=assistant_1, block_id="text-1", chunk="first"),
        events.StreamEnd(message=assistant_1),
        events.ToolCallResult(message=tool_message, results=[]),
        events.HookEvent(
            message=hook_message,
            hook=messages.HookPart(
                hook_id="hook", hook_type="test", status="pending"
            ),
        ),
        events.StreamStart(message=assistant_2),
        events.TextStart(message=assistant_2, block_id="text-2"),
        events.TextDelta(
            message=assistant_2, block_id="text-2", chunk="second"
        ),
        events.StreamEnd(message=assistant_2),
    ]

    hydrated = [hydrator.feed(event) for event in source]

    assert [message.id for message in hydrator.messages] == [
        "assistant-1",
        "tool-message",
        "hook-message",
        "assistant-2",
    ]
    assert list(hydrator.messages_by_id) == [
        "assistant-1",
        "tool-message",
        "hook-message",
        "assistant-2",
    ]
    assert hydrator.messages_by_id["assistant-1"].text == "first"
    assert hydrator.message is hydrator.messages_by_id["assistant-2"]
    assert hydrator.message.text == "second"
    assert hydrated[4] is source[4]
    assert hydrated[5] is source[5]


def test_non_model_events_keep_message() -> None:
    adapter: pydantic.TypeAdapter[events.AgentEvent] = pydantic.TypeAdapter(
        events.OmitEventMessages[events.AgentEvent]
    )
    tool_message = messages.Message(id="tool-message", role="tool", parts=[])
    hook_message = messages.Message(
        id="hook-message", role="internal", parts=[]
    )
    source: list[events.AgentEvent] = [
        events.ToolCallResult(message=tool_message, results=[]),
        events.HookEvent(
            message=hook_message,
            hook=messages.HookPart(
                hook_id="hook", hook_type="test", status="pending"
            ),
        ),
    ]

    dumped = [adapter.dump_python(event) for event in source]

    assert [item["message"]["id"] for item in dumped] == [
        "tool-message",
        "hook-message",
    ]


def test_omitted_model_event_validates_with_dummy_message() -> None:
    compact: pydantic.TypeAdapter[events.AgentEvent] = pydantic.TypeAdapter(
        events.OmitEventMessages[events.AgentEvent]
    )
    regular: pydantic.TypeAdapter[events.AgentEvent] = pydantic.TypeAdapter(
        events.AgentEvent
    )

    event = events.TextDelta(chunk="hello")
    restored = regular.validate_python(compact.dump_python(event))

    assert isinstance(restored, events.TextDelta)
    assert restored.message.id == event.message.id
    assert restored.message.parts == []


class TestReplayMessageEvents:
    async def test_reasoning_signature_survives_replay(self) -> None:
        """A signed reasoning part replayed through the Stream aggregator
        must keep its provider_metadata -- otherwise a rebuilt turn can't
        be replayed to the provider."""
        original = messages.Message(
            role="assistant",
            parts=[
                messages.ReasoningPart(
                    text="thinking hard",
                    provider_metadata={"anthropic": {"signature": "ErMJsig=="}},
                ),
                messages.TextPart(text="the answer is 42"),
            ],
        )

        async with models.Stream(
            events._replay_message_events(original)
        ) as stream:
            async for _ in stream:
                pass

        reasoning = [
            p
            for p in stream.message.parts
            if isinstance(p, messages.ReasoningPart)
        ]
        assert len(reasoning) == 1
        assert reasoning[0].text == "thinking hard"
        assert reasoning[0].provider_metadata == {
            "anthropic": {"signature": "ErMJsig=="}
        }

    async def test_reasoning_signature_on_end_event(self) -> None:
        """The signature rides on the ReasoningEnd event, mirroring how the
        real streaming adapters emit it."""
        msg = messages.Message(
            role="assistant",
            parts=[
                messages.ReasoningPart(
                    text="hmm",
                    provider_metadata={"anthropic": {"signature": "sig"}},
                )
            ],
        )

        reasoning_ends = [
            e
            async for e in events._replay_message_events(msg)
            if isinstance(e, events.ReasoningEnd)
        ]
        assert len(reasoning_ends) == 1
        assert reasoning_ends[0].provider_metadata == {
            "anthropic": {"signature": "sig"}
        }

    async def test_provider_metadata_survives_replay(self) -> None:
        """provider_metadata on every part, and the message itself, round-
        trips through the aggregator -- not just reasoning signatures."""
        original = messages.Message(
            role="assistant",
            parts=[
                messages.TextPart(text="hi", provider_metadata={"p": {"t": 1}}),
                messages.ToolCallPart(
                    tool_call_id="tc-1",
                    tool_name="weather",
                    tool_args="{}",
                    provider_metadata={"p": {"tc": 2}},
                ),
            ],
            provider_metadata={"p": {"msg": 3}},
        )

        async with models.Stream(
            events._replay_message_events(original)
        ) as stream:
            async for _ in stream:
                pass

        rebuilt = stream.message
        assert rebuilt.provider_metadata == {"p": {"msg": 3}}
        text = next(
            p for p in rebuilt.parts if isinstance(p, messages.TextPart)
        )
        assert text.provider_metadata == {"p": {"t": 1}}
        tool = next(
            p for p in rebuilt.parts if isinstance(p, messages.ToolCallPart)
        )
        assert tool.provider_metadata == {"p": {"tc": 2}}
