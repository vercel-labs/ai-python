"""Focused tests for message builder helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from ai.types import builders, messages


def test_user_message_mixed_content() -> None:
    fp = messages.FilePart(
        data="https://example.com/img.png", media_type="image/png"
    )
    msg = builders.user_message("Describe this:", fp, "Thanks")
    assert len(msg.parts) == 3
    assert isinstance(msg.parts[0], messages.TextPart)
    assert isinstance(msg.parts[1], messages.FilePart)
    assert isinstance(msg.parts[2], messages.TextPart)


def test_text_part() -> None:
    tp = builders.text_part("hello", provider_metadata={"k": "v"})
    assert isinstance(tp, messages.TextPart)
    assert tp.text == "hello"
    assert tp.provider_metadata == {"k": "v"}


def test_content_output_coerces_strings() -> None:
    fp = messages.FilePart(
        data="https://example.com/img.png", media_type="image/png"
    )
    out = builders.content_output("Here:", fp)
    assert isinstance(out, messages.ContentOutput)
    assert len(out.value) == 2
    assert isinstance(out.value[0], messages.TextPart)
    assert out.value[0].text == "Here:"
    assert isinstance(out.value[1], messages.FilePart)


def test_file_part_from_url() -> None:
    fp = builders.file_part("https://example.com/image.png")
    assert isinstance(fp, messages.FilePart)
    assert fp.data == "https://example.com/image.png"
    assert fp.media_type == "image/png"


def test_file_part_from_bytes_auto_detect_png() -> None:
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    fp = builders.file_part(png_header, filename="cat.png")
    assert fp.media_type == "image/png"
    assert fp.filename == "cat.png"


def test_file_part_from_bytes_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot detect media_type"):
        builders.file_part(b"\x00\x00\x00")


def test_tool_message_merges_tool_messages() -> None:
    m1 = messages.Message(
        role="tool",
        parts=[builders.tool_result_part("tc-1", result=1, tool_name="a")],
    )
    m2 = messages.Message(
        role="tool",
        parts=[builders.tool_result_part("tc-2", result=2, tool_name="b")],
    )

    merged = builders.tool_message(m1, m2)

    assert merged.role == "tool"
    assert [part.tool_call_id for part in merged.tool_results] == [
        "tc-1",
        "tc-2",
    ]


def test_tool_message_rejects_non_tool_message() -> None:
    with pytest.raises(TypeError, match="Expected tool message"):
        builders.tool_message(builders.user_message("hello"))


def test_tool_message_rejects_non_result_parts() -> None:
    invalid = messages.Message(
        role="tool", parts=[messages.TextPart(text="bad")]
    )
    with pytest.raises(TypeError, match="ToolResultPart"):
        builders.tool_message(invalid)


def test_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        builders.user_message(cast(Any, 42))
