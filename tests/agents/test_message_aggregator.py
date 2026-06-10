"""MessageAggregator — deduping message snapshots by id."""

from __future__ import annotations

import ai
from ai.types import events as events_

from ..conftest import text_msg, tool_result_msg


def test_consecutive_snapshots_replace() -> None:
    """Later snapshots of the same message replace the earlier ones."""
    agg = ai.agents.MessageAggregator()
    agg.feed(events_.StreamEnd(message=text_msg("partial", id="msg-a")))
    agg.feed(events_.StreamEnd(message=text_msg("complete", id="msg-a")))

    bundle = agg.snapshot()
    assert [m.text for m in bundle.messages] == ["complete"]


def test_interleaved_snapshots_replace() -> None:
    """Snapshots of the same message dedupe even when another message
    lands in between (e.g. a tool-result message mid-stream)."""
    agg = ai.agents.MessageAggregator()
    tool = tool_result_msg(tc_id="tc-1", result="r")
    agg.feed(events_.StreamEnd(message=text_msg("partial", id="msg-a")))
    agg.feed(events_.StreamEnd(message=tool))
    agg.feed(events_.StreamEnd(message=text_msg("complete", id="msg-a")))

    bundle = agg.snapshot()
    assert [m.id for m in bundle.messages] == ["msg-a", tool.id]
    assert bundle.messages[0].text == "complete"


def test_first_occurrence_position_is_kept() -> None:
    """Replacement keeps the message at its original position."""
    agg = ai.agents.MessageAggregator()
    tool = tool_result_msg(tc_id="tc-1", result="r")
    agg.feed(events_.StreamEnd(message=text_msg("a1", id="msg-a")))
    agg.feed(events_.StreamEnd(message=tool))
    agg.feed(events_.StreamEnd(message=text_msg("b1", id="msg-b")))
    agg.feed(events_.StreamEnd(message=text_msg("a2", id="msg-a")))
    agg.feed(events_.StreamEnd(message=text_msg("b2", id="msg-b")))

    bundle = agg.snapshot()
    assert [m.id for m in bundle.messages] == ["msg-a", tool.id, "msg-b"]
    assert [m.text for m in bundle.messages] == ["a2", "", "b2"]
