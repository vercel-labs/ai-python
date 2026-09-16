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


def test_retry_drops_response_and_its_tool_results() -> None:
    """Retry removes the message the model stream was building and every
    message after it; the retried stream may come back under a new id."""
    agg = ai.agents.MessageAggregator()
    tool_a = tool_result_msg(tc_id="tc-a", result="r")
    tool_b = tool_result_msg(tc_id="tc-b", result="stale")
    agg.feed(events_.StreamEnd(message=text_msg("done", id="msg-a")))
    agg.feed(events_.ToolCallResult(message=tool_a, results=[]))
    agg.feed(events_.StreamEnd(message=text_msg("par", id="msg-b")))
    agg.feed(events_.ToolCallResult(message=tool_b, results=[]))
    agg.feed(events_.Retry())

    assert [m.id for m in agg.snapshot().messages] == ["msg-a", tool_a.id]

    agg.feed(events_.StreamEnd(message=text_msg("retried", id="msg-c")))
    agg.feed(events_.StreamEnd(message=text_msg("again", id="msg-a")))

    bundle = agg.snapshot()
    assert [m.id for m in bundle.messages] == ["msg-a", tool_a.id, "msg-c"]
    assert bundle.messages[0].text == "again"


def test_retry_without_stream_in_progress_is_noop() -> None:
    agg = ai.agents.MessageAggregator()
    agg.feed(events_.Retry())
    assert agg.snapshot().messages == ()

    tool = tool_result_msg(tc_id="tc-1", result="r")
    agg.feed(events_.StreamEnd(message=text_msg("done", id="msg-a")))
    agg.feed(events_.Retry())
    agg.feed(events_.ToolCallResult(message=tool, results=[]))
    agg.feed(events_.Retry())

    assert [m.id for m in agg.snapshot().messages] == [tool.id]
