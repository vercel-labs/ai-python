"""Focused tests for message model branches with real logic."""

from __future__ import annotations

import pydantic
import pytest

from ai.types import messages


class _Weather(pydantic.BaseModel):
    city: str
    temperature: float


_WEATHER_DATA = {"city": "SF", "temperature": 62.0}
_WEATHER_TYPE_NAME = f"{_Weather.__module__}.{_Weather.__qualname__}"


def test_replace() -> None:
    old_text = messages.TextPart(id="p0", text="hello")
    m = messages.Message(
        id="m1",
        role="assistant",
        parts=[old_text, messages.TextPart(id="p1", text="world")],
    )
    new_text = messages.TextPart(id="p0", text="updated")
    updated_m = m.replace(new_text)
    assert updated_m.parts[0].text == "updated"  # type: ignore[union-attr]
    assert m.parts[0].text == "hello"  # type: ignore[union-attr]


def test_replace_missing_id() -> None:
    m = messages.Message(
        id="m1", role="assistant", parts=[messages.TextPart(id="p0", text="hi")]
    )
    orphan = messages.TextPart(id="no-such-id", text="x")
    with pytest.raises(ValueError, match="in message"):
        m.replace(orphan)


def test_replace_two_arg() -> None:
    old_text = messages.TextPart(id="p0", text="hello")
    m = messages.Message(id="m1", role="assistant", parts=[old_text])
    new_text = messages.TextPart(id="different", text="world")
    updated = m.replace(old_text, new_text)
    part = updated.parts[0]
    assert isinstance(part, messages.TextPart)
    assert part.text == "world"
    assert part.id == "different"
    orig = m.parts[0]
    assert isinstance(orig, messages.TextPart)
    assert orig.text == "hello"


def test_replace_two_arg_missing() -> None:
    m = messages.Message(
        id="m1", role="assistant", parts=[messages.TextPart(id="p0", text="hi")]
    )
    stranger = messages.TextPart(id="p0", text="hi")
    with pytest.raises(ValueError, match="not found in message"):
        m.replace(stranger, messages.TextPart(text="new"))


def test_structured_output_part_value() -> None:
    part = messages.StructuredOutputPart(
        data=_WEATHER_DATA, output_type_name=_WEATHER_TYPE_NAME
    )
    val = part.value
    assert isinstance(val, _Weather)
    assert val.city == "SF"
    assert part.value is val


def test_structured_output_part_bad_class_name() -> None:
    part = messages.StructuredOutputPart(
        data=_WEATHER_DATA, output_type_name="nonexistent.module.Cls"
    )
    with pytest.raises(ImportError):
        _ = part.value


def test_message_output_from_part() -> None:
    m = messages.Message(
        id="m1",
        role="assistant",
        parts=[
            messages.TextPart(text="{}"),
            messages.StructuredOutputPart(
                data=_WEATHER_DATA, output_type_name=_WEATHER_TYPE_NAME
            ),
        ],
    )
    assert isinstance(m.output, _Weather)
    assert m.output.city == "SF"


def test_structured_output_round_trip() -> None:
    m = messages.Message(
        id="m1",
        role="assistant",
        parts=[
            messages.TextPart(text="{}"),
            messages.StructuredOutputPart(
                data=_WEATHER_DATA, output_type_name=_WEATHER_TYPE_NAME
            ),
        ],
    )
    restored = messages.Message.model_validate(m.model_dump())
    assert isinstance(restored.output, _Weather)
    assert restored.output.city == "SF"


def test_usage_add_merges_optional_fields() -> None:
    a = messages.Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=20,
    )
    b = messages.Usage(
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


def test_from_url_infers_from_data_url() -> None:
    fp = messages.FilePart.from_url("data:audio/wav;base64,AAAA")
    assert fp.media_type == "audio/wav"


def test_from_url_explicit_media_type_overrides() -> None:
    fp = messages.FilePart.from_url("https://example.com/img", media_type="image/webp")
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
