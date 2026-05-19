"""Network IO for media — download with size limits and SSRF prevention.

Pure media utilities (detection, encoding, inference) live in
:mod:`ai.types.media`.
"""

from urllib.parse import urlparse

import httpx

DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB (matches TS SDK)
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class DownloadError(Exception):
    """Raised when a URL download fails."""

    def __init__(
        self,
        url: str,
        *,
        status_code: int | None = None,
        status_text: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        parts = [f"Failed to download {url!r}"]
        if status_code is not None:
            parts.append(f"status={status_code}")
        if status_text:
            parts.append(status_text)
        super().__init__(": ".join(parts))
        self.url = url
        self.status_code = status_code
        if cause is not None:
            self.__cause__ = cause


def _validate_url(url: str) -> None:
    """Reject non-HTTP(S) URLs (SSRF prevention)."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise DownloadError(
            url, status_text=f"Unsupported URL scheme: {parsed.scheme!r}"
        )


async def download(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[bytes, str | None]:
    """Download *url* and return ``(data, content_type)``.

    Args:
        url: The URL to fetch (must be ``http`` or ``https``).
        max_bytes: Maximum response size.  Defaults to 100 MiB.

    Returns:
        A tuple of ``(raw_bytes, content_type_or_None)``.

    Raises:
        DownloadError: On any failure (network, HTTP status, size, etc.).

    """
    _validate_url(url)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url)

            # Validate redirect target
            if resp.url is not None and str(resp.url) != url:
                _validate_url(str(resp.url))

            if resp.status_code >= 400:
                raise DownloadError(
                    url,
                    status_code=resp.status_code,
                    status_text=resp.reason_phrase or "",
                )

            data = resp.content
            if len(data) > max_bytes:
                raise DownloadError(
                    url,
                    status_text=(
                        f"Response exceeds maximum size "
                        f"({len(data)} > {max_bytes} bytes)"
                    ),
                )

            content_type = resp.headers.get("content-type")
            # Strip charset/parameters: "image/png; charset=..." → "image/png"
            if content_type:
                content_type = content_type.split(";")[0].strip()

            return data, content_type or None

    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(url, cause=exc) from exc
