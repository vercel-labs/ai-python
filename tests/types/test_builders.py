"""Tests for message builder helpers."""

import pytest

from ai.types.builders import (
    assistant_message,
    file_part,
    system_message,
    thinking,
    tool_message,
    tool_result,
    user_message,
)
from ai.types.messages import (
    FilePart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
)

# -- system_message --------------------------------------------------------


def test_system_message_from_string() -> None:
    msg = system_message("You are helpful.")
    assert msg.role == "system"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], TextPart)
    assert msg.parts[0].text == "You are helpful."


def test_system_message_empty() -> None:
    msg = system_message()
    assert msg.role == "system"
    assert msg.parts == []


# -- user_message ----------------------------------------------------------


def test_user_message_single_string() -> None:
    msg = user_message("Hello")
    assert msg.role == "user"
    assert len(msg.parts) == 1
    assert msg.parts[0].text == "Hello"  # type: ignore[union-attr]


def test_user_message_multiple_strings_stay_separate() -> None:
    msg = user_message("foo", "bar")
    assert len(msg.parts) == 2
    assert msg.parts[0].text == "foo"  # type: ignore[union-attr]
    assert msg.parts[1].text == "bar"  # type: ignore[union-attr]


def test_user_message_mixed_content() -> None:
    fp = FilePart(data="https://example.com/img.png", media_type="image/png")
    msg = user_message("Describe this:", fp, "Thanks")
    assert len(msg.parts) == 3
    assert isinstance(msg.parts[0], TextPart)
    assert isinstance(msg.parts[1], FilePart)
    assert isinstance(msg.parts[2], TextPart)


def test_user_message_part_passthrough() -> None:
    tp = TextPart(text="already a part")
    msg = user_message(tp)
    assert msg.parts[0] is tp


# -- assistant_message -----------------------------------------------------


def test_assistant_message_with_thinking() -> None:
    r = thinking("hmm let me think")
    msg = assistant_message(r, "Here's my answer.")
    assert msg.role == "assistant"
    assert len(msg.parts) == 2
    assert isinstance(msg.parts[0], ReasoningPart)
    assert isinstance(msg.parts[1], TextPart)


def test_assistant_message_with_tool_call_part() -> None:
    tool = ToolCallPart(tool_call_id="tc-1", tool_name="test", tool_args="{}")
    msg = assistant_message("calling tool", tool)
    assert len(msg.parts) == 2
    assert isinstance(msg.parts[1], ToolCallPart)


# -- file_part -------------------------------------------------------------


def test_file_part_from_url() -> None:
    fp = file_part("https://example.com/image.png")
    assert isinstance(fp, FilePart)
    assert fp.data == "https://example.com/image.png"
    assert fp.media_type == "image/png"


def test_file_part_from_bytes_with_explicit_media_type() -> None:
    fp = file_part(b"\x00\x00", media_type="application/octet-stream")
    assert isinstance(fp, FilePart)
    assert fp.media_type == "application/octet-stream"


def test_file_part_from_bytes_auto_detect_png() -> None:
    # PNG magic bytes
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    fp = file_part(png_header)
    assert fp.media_type == "image/png"


def test_file_part_from_bytes_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot detect media_type"):
        file_part(b"\x00\x00\x00")


def test_file_part_with_filename() -> None:
    fp = file_part(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, filename="cat.png")
    assert fp.filename == "cat.png"


# -- thinking --------------------------------------------------------------


def test_thinking_basic() -> None:
    r = thinking("deep thoughts")
    assert isinstance(r, ReasoningPart)
    assert r.text == "deep thoughts"
    assert r.signature is None


def test_thinking_with_signature() -> None:
    r = thinking("deep thoughts", signature="sig123")
    assert r.signature == "sig123"


# -- tool_message ----------------------------------------------------------


def test_tool_message_merges_tool_messages() -> None:
    m1 = Message(
        role="tool",
        parts=[tool_result("tc-1", result=1, tool_name="a")],
    )
    m2 = Message(
        role="tool",
        parts=[tool_result("tc-2", result=2, tool_name="b")],
    )

    merged = tool_message(m1, m2)

    assert merged.role == "tool"
    assert [part.tool_call_id for part in merged.tool_results] == ["tc-1", "tc-2"]


def test_tool_message_accepts_message_list() -> None:
    messages = [
        Message(role="tool", parts=[tool_result("tc-1", result=1, tool_name="a")]),
        Message(role="tool", parts=[tool_result("tc-2", result=2, tool_name="b")]),
    ]

    merged = tool_message(messages)

    assert [part.tool_call_id for part in merged.tool_results] == ["tc-1", "tc-2"]


def test_tool_message_rejects_non_tool_message() -> None:
    with pytest.raises(TypeError, match="Expected tool message"):
        tool_message(user_message("hello"))


def test_tool_message_rejects_non_result_parts() -> None:
    invalid = Message(role="tool", parts=[TextPart(text="bad")])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="ToolResultPart"):
        tool_message(invalid)


# -- type coercion edge cases ----------------------------------------------


def test_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        user_message(42)  # type: ignore[arg-type]
