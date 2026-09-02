"""Tests for ``ai.types.events``."""

from __future__ import annotations

import pydantic

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
