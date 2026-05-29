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
            events.replay_message_events(original)
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
            async for e in events.replay_message_events(msg)
            if isinstance(e, events.ReasoningEnd)
        ]
        assert len(reasoning_ends) == 1
        assert reasoning_ends[0].provider_metadata == {
            "anthropic": {"signature": "sig"}
        }
