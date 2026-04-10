"""Focused tests for media helpers and detection edge cases."""

from __future__ import annotations

import base64

from ai.types import media

_RIFF_WEBP = bytes(
    [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50]
)
_RIFF_WAVE = bytes(
    [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45]
)


def test_data_to_base64_extracts_from_data_url() -> None:
    payload = base64.b64encode(b"hello").decode()
    data_url = f"data:image/png;base64,{payload}"
    assert media.data_to_base64(data_url) == payload


def test_data_to_data_url_from_bytes() -> None:
    raw = b"\x89PNG"
    result = media.data_to_data_url(raw, "image/png")
    assert result.startswith("data:image/png;base64,")
    assert base64.b64decode(result.split(",", 1)[1]) == raw


def test_split_data_url_malformed() -> None:
    assert media.split_data_url("data:nope") == (None, None)


def test_detect_image_media_type_webp_container() -> None:
    assert media.detect_image_media_type(_RIFF_WEBP) == "image/webp"


def test_detect_image_media_type_rejects_wave_container() -> None:
    assert media.detect_image_media_type(_RIFF_WAVE) is None


def test_detect_image_media_type_base64() -> None:
    b64 = base64.b64encode(_RIFF_WEBP).decode()
    assert media.detect_image_media_type(b64) == "image/webp"


def test_detect_audio_media_type_strips_id3() -> None:
    id3_header = bytes([0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    assert media.detect_audio_media_type(id3_header + bytes([0xFF, 0xFB])) == (
        "audio/mpeg"
    )


def test_detect_audio_media_type_rejects_webp_container() -> None:
    assert media.detect_audio_media_type(_RIFF_WEBP) is None


def test_detect_audio_media_type_base64() -> None:
    b64 = base64.b64encode(_RIFF_WAVE).decode()
    assert media.detect_audio_media_type(b64) == "audio/wav"


def test_unknown_media_formats_return_none() -> None:
    unknown = bytes([0x00, 0x01, 0x02, 0x03])
    assert media.detect_image_media_type(unknown) is None
    assert media.detect_audio_media_type(unknown) is None


def test_empty_or_short_media_returns_none() -> None:
    assert media.detect_image_media_type(b"") is None
    assert media.detect_audio_media_type(b"") is None
    assert media.detect_image_media_type(bytes([0x89])) is None
    assert media.detect_audio_media_type(bytes([0xFF])) is None
