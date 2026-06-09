"""Focused tests for message model branches with real logic."""

from __future__ import annotations

from typing import Any

import pytest

from ai.types import messages, usage


def test_usage_add_merges_optional_fields() -> None:
    a = usage.Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=20,
    )
    b = usage.Usage(
        input_tokens=200,
        output_tokens=80,
        reasoning_tokens=10,
    )
    total = a + b

    assert total.input_tokens == 300
    assert total.output_tokens == 130
    assert total.total_tokens == 430
    assert total.reasoning_tokens == 10
    assert total.cache_read_tokens == 20
    assert total.cache_write_tokens is None
    assert total.raw is None


def test_file_part_in_part_union() -> None:
    msg = messages.Message(
        id="m1",
        role="user",
        parts=[
            messages.TextPart(text="look at this"),
            messages.FilePart(
                data="https://example.com/cat.jpg", media_type="image/jpeg"
            ),
        ],
    )
    dumped = msg.model_dump()
    restored = messages.Message.model_validate(dumped)
    assert len(restored.parts) == 2
    assert isinstance(restored.parts[1], messages.FilePart)
    assert restored.parts[1].media_type == "image/jpeg"


def test_provider_metadata_round_trips_as_dict() -> None:
    msg = messages.Message(
        id="m1",
        role="assistant",
        provider_metadata={"provider": "test", "nested": {"value": 1}},
        parts=[
            messages.TextPart(
                text="hello",
                provider_metadata={"provider": "test", "block": "text"},
            )
        ],
    )

    restored = messages.Message.model_validate(msg.model_dump())

    assert restored.provider_metadata == {
        "provider": "test",
        "nested": {"value": 1},
    }
    part = restored.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.provider_metadata == {"provider": "test", "block": "text"}


def test_from_url_infers_from_data_url() -> None:
    fp = messages.FilePart.from_url("data:audio/wav;base64,AAAA")
    assert fp.media_type == "audio/wav"


def test_from_url_explicit_media_type_overrides() -> None:
    fp = messages.FilePart.from_url(
        "https://example.com/img", media_type="image/webp"
    )
    assert fp.media_type == "image/webp"


def test_from_url_unknown_extension_raises() -> None:
    with pytest.raises(ValueError, match="Cannot infer media_type"):
        messages.FilePart.from_url("https://example.com/blob")


def test_from_bytes_detects_image_and_preserves_filename() -> None:
    data = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A])
    fp = messages.FilePart.from_bytes(data, filename="photo.png")
    assert fp.media_type == "image/png"
    assert fp.data == data
    assert fp.filename == "photo.png"


def test_from_bytes_detects_audio() -> None:
    data = bytes(
        [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]
    )
    fp = messages.FilePart.from_bytes(data)
    assert fp.media_type == "audio/wav"


def test_from_bytes_explicit_overrides() -> None:
    fp = messages.FilePart.from_bytes(b"\x00\x00", media_type="video/mp4")
    assert fp.media_type == "video/mp4"


def test_from_bytes_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot detect media_type"):
        messages.FilePart.from_bytes(b"\x00\x01\x02\x03")


# ---------------------------------------------------------------------------
# ToolResultPart -- typed result coercion and round-trip
# ---------------------------------------------------------------------------


def test_tool_result_content_output_with_file_part_round_trip() -> None:
    """FilePart inside ContentOutput survives JSON round-trip."""
    fp = messages.FilePart(data=b"fake-image-data", media_type="image/png")
    trp = messages.ToolResultPart(
        tool_call_id="tc1",
        tool_name="read",
        result=messages.ContentOutput(
            value=[messages.TextPart(text="label"), fp]
        ),
        result_kind="special",
    )
    j = trp.model_dump_json()
    restored = messages.ToolResultPart.model_validate_json(j)
    assert isinstance(restored.result, messages.ContentOutput)
    assert len(restored.result.value) == 2
    text_part, file_part = restored.result.value
    assert isinstance(text_part, messages.TextPart)
    assert text_part.text == "label"
    assert isinstance(file_part, messages.FilePart)
    assert file_part.media_type == "image/png"


def test_tool_result_plain_values_stored_raw() -> None:
    """Plain str / dict / list / None results are stored as-is and round-trip.

    ``result`` is ``Any`` -- there is no wrapper type, so a tool's return
    value lands on the part unchanged and survives a JSON round-trip.
    """
    cases: list[Any] = ["hello", None, [1, 2, 3], {"key": "val"}]
    for raw in cases:
        trp = messages.ToolResultPart(
            tool_call_id="tc", tool_name="t", result=raw
        )
        assert trp.result == raw
        assert trp.result_kind == "json"
        restored = messages.ToolResultPart.model_validate_json(
            trp.model_dump_json()
        )
        assert restored.result == raw
        assert restored.result_kind == "json"


def test_tool_result_content_in_message_round_trip() -> None:
    """ContentOutput with a FilePart survives Message round-trip."""
    fp = messages.FilePart(data=b"img-data", media_type="image/webp")
    msg = messages.Message(
        role="tool",
        parts=[
            messages.ToolResultPart(
                tool_call_id="tc",
                tool_name="read",
                result=messages.ContentOutput(
                    value=[messages.TextPart(text="Read image"), fp]
                ),
                result_kind="special",
            )
        ],
    )
    j = msg.model_dump_json()
    restored = messages.Message.model_validate_json(j)
    part = restored.parts[0]
    assert isinstance(part, messages.ToolResultPart)
    assert isinstance(part.result, messages.ContentOutput)
    fp2 = part.result.value[1]
    assert isinstance(fp2, messages.FilePart)
    assert fp2.media_type == "image/webp"


def test_tool_result_file_part_base64_valid_after_round_trip() -> None:
    """After round-trip, data_to_base64 produces standard base-64."""
    import base64

    from ai.types import media as media_

    raw = b"\xff\xd8\xff\xe0\x00\x10JFIF" * 10
    fp = messages.FilePart(data=raw, media_type="image/jpeg")
    trp = messages.ToolResultPart(
        tool_call_id="tc",
        tool_name="read",
        result=messages.ContentOutput(
            value=[messages.TextPart(text="label"), fp]
        ),
        result_kind="special",
    )
    restored = messages.ToolResultPart.model_validate_json(
        trp.model_dump_json()
    )
    assert isinstance(restored.result, messages.ContentOutput)
    fp2 = restored.result.value[1]
    assert isinstance(fp2, messages.FilePart)

    b64 = media_.data_to_base64(fp2.data)
    assert "_" not in b64
    assert "-" not in b64
    decoded = base64.b64decode(b64)
    assert decoded == raw
