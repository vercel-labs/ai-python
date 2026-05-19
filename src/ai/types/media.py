"""Media type detection, inference, and encoding utilities.

Pure functions with no IO dependencies — safe to use from any layer.
"""

from __future__ import annotations

import base64
import base64 as _b64
import mimetypes

# -- URL helpers -----------------------------------------------------------


def is_url(data: str) -> bool:
    """Return True if *data* looks like a URL rather than raw base-64."""
    return data.startswith(("http://", "https://", "data:"))


def is_downloadable_url(data: str) -> bool:
    """Return True if *data* is an ``http(s)://`` URL that can be fetched."""
    return data.startswith(("http://", "https://"))


def split_data_url(url: str) -> tuple[str | None, str | None]:
    """Parse a ``data:`` URL into ``(media_type, base64_content)``.

    Returns ``(None, None)`` if the input is not a valid ``data:`` URL.

    Example::

        >>> split_data_url("data:image/png;base64,iVBOR...")
        ("image/png", "iVBOR...")
    """
    if not url.startswith("data:"):
        return None, None
    try:
        header, b64_content = url.split(",", 1)
        mt = header.split(";")[0].split(":", 1)[1]
        return (mt or None), (b64_content or None)
    except (ValueError, IndexError):
        return None, None


# -- encoding helpers ------------------------------------------------------


def data_to_base64(data: str | bytes) -> str:
    """Ensure *data* is a base-64 encoded string.

    * ``bytes`` -> base-64 encoded.
    * ``str`` that is a ``data:`` URL -> base-64 content extracted.
    * ``str`` that is an ``http(s)://`` URL -> returned as-is (caller
      must handle).
    * ``str`` that is not a URL -> assumed to already be base-64.
    """
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    if data.startswith("data:"):
        _, b64 = split_data_url(data)
        if b64 is not None:
            return b64
    return data


def data_to_data_url(data: str | bytes, media_type: str) -> str:
    """Convert *data* to a ``data:`` URL.  Passes through existing URLs."""
    if isinstance(data, str) and is_url(data):
        return data
    b64 = data_to_base64(data)
    return f"data:{media_type};base64,{b64}"


# -- media-type inference --------------------------------------------------


def infer_media_type(url: str) -> str:
    """Infer IANA media type from a URL.

    * ``data:image/png;base64,...`` -> ``"image/png"``
    * ``https://example.com/cat.jpg`` -> ``"image/jpeg"`` (via :mod:`mimetypes`)
    * Unknown -> raises :class:`ValueError`
    """
    if url.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        rest = url[5:]  # strip "data:"
        sep = rest.find(",")
        meta = rest[:sep] if sep != -1 else rest
        mt = meta.split(";")[0]
        if mt:
            return mt
    else:
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            return guessed
    raise ValueError(
        f"Cannot infer media_type from URL: {url!r}. "
        "Provide media_type explicitly."
    )


# ---------------------------------------------------------------------------
# Signature definitions
# ---------------------------------------------------------------------------

# Each signature is a tuple of (media_type, byte_prefix) where byte_prefix
# is a tuple of ``int | None`` values.  ``None`` is a wildcard that matches
# any byte (mirrors the TS SDK's ``null`` sentinel).

_Signature = tuple[str, tuple[int | None, ...]]

IMAGE_SIGNATURES: list[_Signature] = [
    ("image/gif", (0x47, 0x49, 0x46)),
    ("image/png", (0x89, 0x50, 0x4E, 0x47)),
    ("image/jpeg", (0xFF, 0xD8)),
    (
        "image/webp",
        (
            0x52,
            0x49,
            0x46,
            0x46,
            None,
            None,
            None,
            None,
            0x57,
            0x45,
            0x42,
            0x50,
        ),
    ),
    ("image/bmp", (0x42, 0x4D)),
    ("image/tiff", (0x49, 0x49, 0x2A, 0x00)),  # little-endian
    ("image/tiff", (0x4D, 0x4D, 0x00, 0x2A)),  # big-endian
    (
        "image/avif",
        (
            0x00,
            0x00,
            0x00,
            0x20,
            0x66,
            0x74,
            0x79,
            0x70,
            0x61,
            0x76,
            0x69,
            0x66,
        ),
    ),
    (
        "image/heic",
        (
            0x00,
            0x00,
            0x00,
            0x20,
            0x66,
            0x74,
            0x79,
            0x70,
            0x68,
            0x65,
            0x69,
            0x63,
        ),
    ),
]

AUDIO_SIGNATURES: list[_Signature] = [
    ("audio/mpeg", (0xFF, 0xFB)),
    ("audio/mpeg", (0xFF, 0xFA)),
    ("audio/mpeg", (0xFF, 0xF3)),
    ("audio/mpeg", (0xFF, 0xF2)),
    ("audio/mpeg", (0xFF, 0xE3)),
    ("audio/mpeg", (0xFF, 0xE2)),
    (
        "audio/wav",
        (
            0x52,
            0x49,
            0x46,
            0x46,
            None,
            None,
            None,
            None,
            0x57,
            0x41,
            0x56,
            0x45,
        ),
    ),
    ("audio/ogg", (0x4F, 0x67, 0x67, 0x53)),
    ("audio/flac", (0x66, 0x4C, 0x61, 0x43)),
    ("audio/aac", (0x40, 0x15, 0x00, 0x00)),
    ("audio/mp4", (0x66, 0x74, 0x79, 0x70)),
    ("audio/webm", (0x1A, 0x45, 0xDF, 0xA3)),
]

VIDEO_SIGNATURES: list[_Signature] = [
    ("video/mp4", (0x00, 0x00, 0x00, None, 0x66, 0x74, 0x79, 0x70)),
    ("video/webm", (0x1A, 0x45, 0xDF, 0xA3)),
    (
        "video/quicktime",
        (0x00, 0x00, 0x00, 0x14, 0x66, 0x74, 0x79, 0x70, 0x71, 0x74),
    ),
    ("video/x-msvideo", (0x52, 0x49, 0x46, 0x46)),
]


# ---------------------------------------------------------------------------
# ID3 tag stripping  (for MP3 files that start with ID3v2 metadata)
# ---------------------------------------------------------------------------

_ID3_HEADER = bytes([0x49, 0x44, 0x33])  # "ID3"
_ID3_BASE64 = "SUQz"  # base64("ID3")


def _strip_id3_tags(data: bytes) -> bytes:
    """Strip an ID3v2 tag header if present, returning the audio data."""
    if len(data) < 10 or data[:3] != _ID3_HEADER:
        return data
    # Syncsafe integer: 4 bytes, 7 bits each
    size = (
        (data[6] & 0x7F) << 21
        | (data[7] & 0x7F) << 14
        | (data[8] & 0x7F) << 7
        | (data[9] & 0x7F)
    )
    offset = size + 10
    return data[offset:] if offset < len(data) else data


def _strip_id3_tags_base64(data: str) -> str:
    """Strip an ID3v2 tag from base64-encoded data if present."""
    if not data.startswith(_ID3_BASE64):
        return data
    # Decode enough to read the ID3 header (10 bytes = ~16 base64 chars)
    try:
        header = _b64.b64decode(data[:16])
    except Exception:
        return data
    if len(header) < 10 or header[:3] != _ID3_HEADER:
        return data
    size = (
        (header[6] & 0x7F) << 21
        | (header[7] & 0x7F) << 14
        | (header[8] & 0x7F) << 7
        | (header[9] & 0x7F)
    )
    offset = size + 10
    # Re-encode: decode full data, strip, re-encode
    try:
        full = _b64.b64decode(data)
        stripped = full[offset:] if offset < len(full) else full
        return _b64.b64encode(stripped).decode("ascii")
    except Exception:
        return data


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def _to_bytes(data: bytes | str, *, max_bytes: int = 24) -> bytes:
    """Convert *data* to bytes for signature comparison.

    For ``str`` input (base-64), decodes only the first *max_bytes*
    characters worth of data to avoid decoding large payloads.
    """
    if isinstance(data, bytes):
        return data[:max_bytes]
    # base-64: 4 chars → 3 bytes.  Decode ~32 chars to get enough bytes.
    chunk = data[: max_bytes * 2]
    # Pad to multiple of 4 for valid base64
    padded = chunk + "=" * (-len(chunk) % 4)
    try:
        return _b64.b64decode(padded)[:max_bytes]
    except Exception:
        return b""


def detect_media_type(
    data: bytes | str,
    signatures: list[_Signature],
) -> str | None:
    """Detect media type from magic bytes.

    Args:
        data: Raw bytes or a base-64 encoded string.
        signatures: List of ``(media_type, byte_prefix)`` tuples to
            match against (e.g. :data:`IMAGE_SIGNATURES`).

    Returns:
        The matched IANA media type, or ``None`` if no signature matches.

    """
    # Strip ID3 tags for audio detection
    if signatures is AUDIO_SIGNATURES:
        if isinstance(data, bytes):
            data = _strip_id3_tags(data)
        else:
            data = _strip_id3_tags_base64(data)

    raw = _to_bytes(data)
    if not raw:
        return None

    for media_type, prefix in signatures:
        if len(raw) < len(prefix):
            continue
        if all(
            expected is None or raw[i] == expected
            for i, expected in enumerate(prefix)
        ):
            return media_type

    return None


def detect_image_media_type(data: bytes | str) -> str | None:
    """Detect image format from magic bytes."""
    return detect_media_type(data, IMAGE_SIGNATURES)


def detect_audio_media_type(data: bytes | str) -> str | None:
    """Detect audio format from magic bytes."""
    return detect_media_type(data, AUDIO_SIGNATURES)
