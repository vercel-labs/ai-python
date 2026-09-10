"""Download media with size limits and private-network SSRF protection.

SDK-managed downloads validate every redirect and pin connections to checked
DNS results. Private media URLs and environment proxies are not supported;
applications that need trusted internal media can fetch it and pass bytes.
This policy does not apply to configured model endpoints or provider-side
fetches. Network-specific address translation still needs egress controls.

Pure media utilities (detection, encoding, inference) live in
:mod:`ai.types.media`.
"""

import asyncio
import ipaddress
import socket

import httpx2 as httpx

DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB
_MAX_REDIRECTS = 10
_DNS_TIMEOUT = 5.0
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")
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


def _validate_address(address: str, url: str) -> None:
    """Allow only public unicast addresses, including embedded IPv4 targets."""
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address):
        if (
            ip.scope_id is not None
            or ip.is_site_local
            or ip.sixtofour
            or ip.teredo
        ):
            raise DownloadError(url, status_text="Non-public IP address")
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if not ip.is_global or ip.is_multicast or ip.is_reserved:
        raise DownloadError(url, status_text="Non-public IP address")


def _validate_url(url: httpx.URL) -> None:
    """Use the HTTP client's parser so validation and requests agree."""
    if url.scheme not in _ALLOWED_SCHEMES:
        raise DownloadError(
            str(url), status_text=f"Unsupported URL scheme: {url.scheme!r}"
        )
    host = url.host.lower().rstrip(".")
    if (
        not host
        or url.userinfo
        or "%" in host
        or host == "localhost"
        or host.endswith((".localhost", ".local"))
    ):
        raise DownloadError(str(url), status_text="Disallowed download URL")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        # Names and alternate IPv4 spellings are checked after resolution.
        return
    _validate_address(host, str(url))


class _DownloadTransport(httpx.AsyncHTTPTransport):
    def __init__(self) -> None:
        super().__init__(
            trust_env=False,
            # Pooling by the pinned IP could reuse another hostname's TLS
            # session. A fresh connection per hop preserves TLS verification.
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        url = request.url
        _validate_url(url)
        async with asyncio.timeout(_DNS_TIMEOUT):
            records = await asyncio.get_running_loop().getaddrinfo(
                url.raw_host.decode("ascii"),
                url.port or (443 if url.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        addresses = list(dict.fromkeys(record[4][0] for record in records))
        if not addresses:
            raise DownloadError(
                str(url), status_text="Hostname has no addresses"
            )
        # Reject the entire DNS answer before connecting, not just the first IP.
        for address in addresses:
            _validate_address(address, str(url))
        for index, address in enumerate(addresses):
            pinned = httpx.Request(
                request.method,
                url.copy_with(host=address),
                headers=request.headers,
                stream=request.stream,
                extensions={
                    **request.extensions,
                    "sni_hostname": url.raw_host.decode("ascii"),
                },
            )
            # Host is generated from the original URL, never the numeric IP.
            pinned.headers["Host"] = url.netloc.decode("ascii")
            try:
                return await super().handle_async_request(pinned)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if index == len(addresses) - 1:
                    raise
        raise AssertionError("No download address attempted")


async def download(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[bytes, str | None]:
    """Download *url* and return ``(data, content_type)``.

    Args:
        url: A public ``http`` or ``https`` URL. Private destinations, including
            through DNS or redirects, and URL credentials are rejected.
        max_bytes: Maximum response size.  Defaults to 100 MiB.

    Returns:
        A tuple of ``(raw_bytes, content_type_or_None)``.

    Raises:
        DownloadError: On any failure (network, HTTP status, size, etc.).

    """
    try:
        if max_bytes < 0:
            raise DownloadError(
                url, status_text="max_bytes must be non-negative"
            )
        current = httpx.URL(url)
        async with httpx.AsyncClient(
            transport=_DownloadTransport(),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            for redirect_count in range(_MAX_REDIRECTS + 1):
                async with client.stream("GET", current) as resp:
                    if resp.has_redirect_location:
                        if redirect_count == _MAX_REDIRECTS:
                            raise DownloadError(
                                url, status_text="Too many redirects"
                            )
                        current = current.join(resp.headers["location"])
                        continue
                    if not resp.is_success:
                        raise DownloadError(
                            url,
                            status_code=resp.status_code,
                            status_text=resp.reason_phrase or "",
                        )
                    length = resp.headers.get("content-length")
                    if (
                        length
                        and length.isdecimal()
                        and int(length) > max_bytes
                    ):
                        raise DownloadError(
                            url, status_text="Response exceeds maximum size"
                        )
                    data = bytearray()
                    async for chunk in resp.aiter_bytes():
                        if len(data) + len(chunk) > max_bytes:
                            raise DownloadError(
                                url, status_text="Response exceeds maximum size"
                            )
                        data.extend(chunk)
                    content_type = resp.headers.get("content-type")
                    if content_type:
                        content_type = content_type.split(";")[0].strip()
                    return bytes(data), content_type or None
        raise AssertionError("Redirect limit not enforced")

    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(url, cause=exc) from exc
