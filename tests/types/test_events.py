"""Tests for ``ai.types.events``."""

from __future__ import annotations

from ai import models
from ai.types import events, messages


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
