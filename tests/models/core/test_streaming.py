"""StreamHandler: event accumulation, state transitions, message building."""

from __future__ import annotations

from ai.models.core.helpers import streaming
from ai.types import messages

# -- Text streaming --------------------------------------------------------


def test_text_lifecycle() -> None:
    h = streaming.StreamHandler(message_id="m1")
    m = h.handle_event(streaming.TextStart(block_id="b1"))
    assert len(m.parts) == 1
    part = m.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.state == "streaming"
    assert part.text == ""

    m = h.handle_event(streaming.TextDelta(block_id="b1", delta="Hello"))
    part = m.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.text == "Hello"
    assert part.delta == "Hello"
    assert part.state == "streaming"

    m = h.handle_event(streaming.TextDelta(block_id="b1", delta=" world"))
    part = m.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.text == "Hello world"
    assert part.delta == " world"

    m = h.handle_event(streaming.TextEnd(block_id="b1"))
    part = m.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.state == "done"
    assert part.delta is None


# -- Reasoning streaming ---------------------------------------------------


def test_reasoning_lifecycle() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.ReasoningStart(block_id="r1"))
    m = h.handle_event(streaming.ReasoningDelta(block_id="r1", delta="thinking"))
    part = m.parts[0]
    assert isinstance(part, messages.ReasoningPart)
    assert part.text == "thinking"
    assert part.state == "streaming"

    m = h.handle_event(streaming.ReasoningEnd(block_id="r1", signature="sig123"))
    part = m.parts[0]
    assert isinstance(part, messages.ReasoningPart)
    assert part.state == "done"
    assert part.signature == "sig123"


# -- Tool streaming --------------------------------------------------------


def test_tool_lifecycle() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.ToolStart(tool_call_id="tc1", tool_name="get_weather"))
    m = h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc1", delta='{"ci'))
    part = m.parts[0]
    assert isinstance(part, messages.ToolCallPart)
    assert part.tool_name == "get_weather"
    assert part.tool_args == '{"ci'
    assert part.state == "streaming"
    assert part.args_delta == '{"ci'

    m = h.handle_event(
        streaming.ToolArgsDelta(tool_call_id="tc1", delta='ty":"London"}')
    )
    part = m.parts[0]
    assert isinstance(part, messages.ToolCallPart)
    assert part.tool_args == '{"city":"London"}'

    m = h.handle_event(streaming.ToolEnd(tool_call_id="tc1"))
    part = m.parts[0]
    assert isinstance(part, messages.ToolCallPart)
    assert part.state == "done"
    assert part.args_delta is None


# -- Multi-part messages ---------------------------------------------------


def test_reasoning_then_text_then_tool() -> None:
    """Full message: reasoning block, text block, tool call."""
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.ReasoningStart(block_id="r1"))
    h.handle_event(streaming.ReasoningDelta(block_id="r1", delta="Let me think"))
    h.handle_event(streaming.ReasoningEnd(block_id="r1"))

    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="I'll check"))
    h.handle_event(streaming.TextEnd(block_id="t1"))

    h.handle_event(streaming.ToolStart(tool_call_id="tc1", tool_name="search"))
    h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc1", delta='{"q":"test"}'))
    m = h.handle_event(streaming.ToolEnd(tool_call_id="tc1"))

    assert len(m.parts) == 3
    assert isinstance(m.parts[0], messages.ReasoningPart)
    assert isinstance(m.parts[1], messages.TextPart)
    assert isinstance(m.parts[2], messages.ToolCallPart)
    assert all(
        p.state == "done"
        for p in m.parts
        if isinstance(
            p, (messages.TextPart, messages.ToolCallPart, messages.ReasoningPart)
        )
    )


def test_multiple_tool_calls() -> None:
    """Parallel tool calls in one message."""
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.ToolStart(tool_call_id="tc1", tool_name="read_file"))
    h.handle_event(streaming.ToolStart(tool_call_id="tc2", tool_name="list_files"))

    m = h.handle_event(
        streaming.ToolArgsDelta(tool_call_id="tc1", delta='{"path":"a.py"}')
    )
    # Both tools should be in parts
    tool_parts = [p for p in m.parts if isinstance(p, messages.ToolCallPart)]
    assert len(tool_parts) == 2
    # tc1 has args, tc2 is empty
    assert tool_parts[0].tool_args == '{"path":"a.py"}'
    assert tool_parts[1].tool_args == ""

    h.handle_event(streaming.ToolArgsDelta(tool_call_id="tc2", delta='{"dir":"."}'))
    h.handle_event(streaming.ToolEnd(tool_call_id="tc1"))
    m = h.handle_event(streaming.ToolEnd(tool_call_id="tc2"))
    assert all(
        p.state == "done"
        for p in m.parts
        if isinstance(
            p, (messages.TextPart, messages.ToolCallPart, messages.ReasoningPart)
        )
    )


# -- MessageDone -----------------------------------------------------------


def test_message_done_finalizes_all() -> None:
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="hello"))
    # Don't send TextEnd -- MessageDone should finalize everything
    m = h.handle_event(streaming.MessageDone(finish_reason="end_turn"))
    part = m.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.state == "done"
    assert m.is_done


def test_message_done_propagates_usage() -> None:
    """Usage on MessageDone surfaces on the built Message."""
    usage = messages.Usage(input_tokens=10, output_tokens=20)
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="hi"))

    # Before MessageDone, usage should not be on the message
    m = h.handle_event(streaming.TextEnd(block_id="t1"))
    assert m.usage is None

    m = h.handle_event(streaming.MessageDone(usage=usage))
    assert m.usage is not None
    assert m.usage.input_tokens == 10
    assert m.usage.output_tokens == 20
    assert m.usage.total_tokens == 30


# -- Message properties propagate ------------------------------------------


def test_deltas_only_on_active_blocks() -> None:
    """Delta should be None on inactive blocks, present only on active."""
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="first"))
    h.handle_event(streaming.TextEnd(block_id="t1"))

    h.handle_event(streaming.TextStart(block_id="t2"))
    m = h.handle_event(streaming.TextDelta(block_id="t2", delta="second"))

    text_parts = [p for p in m.parts if isinstance(p, messages.TextPart)]
    assert text_parts[0].delta is None  # t1 is done
    assert text_parts[1].delta == "second"  # t2 is active


# -- File event (inline images from LLMs like Gemini/GPT-5) ---------------


def test_file_event_accumulates() -> None:
    """FileEvent should produce a FilePart in the message."""
    h = streaming.StreamHandler(message_id="m1")
    m = h.handle_event(
        streaming.FileEvent(block_id="f1", media_type="image/png", data="iVBORw0KGgo=")
    )
    file_parts = [p for p in m.parts if isinstance(p, messages.FilePart)]
    assert len(file_parts) == 1
    assert file_parts[0].media_type == "image/png"
    assert file_parts[0].data == "iVBORw0KGgo="


def test_file_event_with_text() -> None:
    """A message can have both text and file parts (e.g. Gemini image gen)."""
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(streaming.TextStart(block_id="t1"))
    h.handle_event(streaming.TextDelta(block_id="t1", delta="Here is your image:"))
    h.handle_event(streaming.TextEnd(block_id="t1"))
    h.handle_event(
        streaming.FileEvent(block_id="f1", media_type="image/png", data="iVBORw0KGgo=")
    )
    m = h.handle_event(streaming.MessageDone(finish_reason="stop"))

    assert len(m.parts) == 2
    assert isinstance(m.parts[0], messages.TextPart)
    assert m.parts[0].text == "Here is your image:"
    assert isinstance(m.parts[1], messages.FilePart)
    assert m.parts[1].media_type == "image/png"
    assert m.is_done


def test_multiple_file_events() -> None:
    """Multiple FileEvents produce multiple FileParts."""
    h = streaming.StreamHandler(message_id="m1")
    h.handle_event(
        streaming.FileEvent(block_id="f1", media_type="image/png", data="png_data")
    )
    m = h.handle_event(
        streaming.FileEvent(block_id="f2", media_type="image/jpeg", data="jpeg_data")
    )
    file_parts = [p for p in m.parts if isinstance(p, messages.FilePart)]
    assert len(file_parts) == 2
    assert file_parts[0].media_type == "image/png"
    assert file_parts[1].media_type == "image/jpeg"
