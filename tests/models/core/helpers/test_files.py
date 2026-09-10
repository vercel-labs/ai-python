from __future__ import annotations

import asyncio
import gzip
import socket
import ssl
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpcore2
import httpx2 as httpx
import pytest

from ai.models.core.helpers import files
from ai.providers.ai_gateway.protocol import v3, v4
from ai.providers.openai import protocol as openai_protocol
from ai.types import messages

PUBLIC_IP = "93.184.216.34"
PUBLIC_URL = "https://media.example/file?signature=a%2Fb"


@pytest.fixture
async def network() -> AsyncIterator[Any]:
    """Keep validation real; mock only DNS and the underlying HTTP transport."""
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]
    with (
        mock.patch.object(
            asyncio.get_running_loop(), "getaddrinfo", return_value=records
        ) as dns,
        mock.patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            return_value=httpx.Response(200, content=b"media"),
        ) as send,
    ):
        yield dns, send


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "data:text/plain,hello",
        "http:///missing-host",
        "http://[broken/",
        "http://user:password@example.com/",
        "http://localhost/",
        "http://LOCALHOST./",
        "http://sub.localhost/",
        "http://service.local/",
        "http://127.0.0.1/",
        "http://127.255.255.255/",
        "http://0.0.0.0/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/",
        "http://192.0.2.1/",
        "http://198.18.0.1/",
        "http://224.0.0.1/",
        "http://255.255.255.255/",
        "http://[::]/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "http://[fe80::1%25en0]/",
        "http://[fec0::1]/",
        "http://[ff02::1]/",
        "http://[2001:db8::1]/",
        "http://[3fff::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:0:127.0.0.1]/",
        "http://[::127.0.0.1]/",
        "http://[64:ff9b::169.254.169.254]/",
        "http://[64:ff9b:1::127.0.0.1]/",
        "http://[2002:7f00:1::]/",
    ],
)
async def test_reject_unsafe_urls_before_network(
    url: str, network: Any
) -> None:
    dns, send = network
    with pytest.raises(files.DownloadError):
        await files.download(url)
    dns.assert_not_called()
    send.assert_not_called()


@pytest.mark.parametrize("addresses", [["127.0.0.1"], [PUBLIC_IP, "::1"], []])
async def test_reject_private_or_empty_dns(
    addresses: list[str], network: Any
) -> None:
    dns, send = network
    dns.return_value = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
        for address in addresses
    ]
    with pytest.raises(files.DownloadError):
        await files.download(PUBLIC_URL)
    send.assert_not_called()


@pytest.mark.parametrize(
    "host", ["2130706433", "127.1", "0177.0.0.1", "0x7f000001"]
)
async def test_alternate_ipv4_uses_real_resolver(host: str) -> None:
    with mock.patch.object(
        httpx.AsyncHTTPTransport, "handle_async_request"
    ) as send:
        with pytest.raises(files.DownloadError):
            await files.download(f"http://{host}/internal")
        send.assert_not_called()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "target",
    ["http://127.0.0.1/internal", "http://[::1]/", "file:///etc/passwd"],
)
async def test_private_redirect_never_requested(
    status: int, target: str, network: Any
) -> None:
    dns, send = network
    send.return_value = httpx.Response(status, headers={"location": target})
    with pytest.raises(files.DownloadError):
        await files.download(PUBLIC_URL)
    assert dns.call_count == 1
    assert send.call_count == 1


async def test_dns_rebinding_on_redirect(network: Any) -> None:
    dns, send = network
    dns.side_effect = [
        dns.return_value,
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ]
    send.return_value = httpx.Response(302, headers={"location": "/internal"})
    with pytest.raises(files.DownloadError, match="Non-public IP"):
        await files.download(PUBLIC_URL)
    assert dns.call_count == 2
    assert send.call_count == 1


async def test_public_redirect_and_pinning(network: Any) -> None:
    dns, send = network
    send.side_effect = [
        httpx.Response(302, headers={"location": "/final?signature=x%2Fy"}),
        httpx.Response(
            200,
            content=b"image",
            headers={"content-type": "image/png; charset=utf-8"},
        ),
    ]
    assert await files.download(PUBLIC_URL) == (b"image", "image/png")
    assert dns.call_count == 2
    first, second = [call.args[0] for call in send.call_args_list]
    assert first.url.host == PUBLIC_IP
    assert first.url.raw_path == b"/file?signature=a%2Fb"
    assert second.url.raw_path == b"/final?signature=x%2Fy"
    assert second.headers["host"] == "media.example"
    assert second.extensions["sni_hostname"] == "media.example"


async def test_public_address_fallback(network: Any) -> None:
    dns, send = network
    dns.return_value.insert(
        0,
        (
            socket.AF_INET6,
            socket.SOCK_STREAM,
            6,
            "",
            ("2606:4700:4700::1111", 443, 0, 0),
        ),
    )
    send.side_effect = [
        httpx.ConnectError("IPv6 unavailable"),
        httpx.Response(200, content=b"ok"),
    ]
    assert await files.download(PUBLIC_URL) == (b"ok", None)
    assert [call.args[0].url.host for call in send.call_args_list] == [
        "2606:4700:4700::1111",
        PUBLIC_IP,
    ]


async def test_redirect_limit(network: Any) -> None:
    _, send = network
    send.side_effect = lambda _: httpx.Response(
        302, headers={"location": "/again"}
    )
    with pytest.raises(files.DownloadError, match="Too many redirects"):
        await files.download(PUBLIC_URL)
    assert send.call_count == 11


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("headers", [{}, {"content-length": "1"}])
async def test_streaming_size_limit(
    headers: dict[str, str], network: Any
) -> None:
    _, send = network
    body = Body([b"123", b"456", b"never read"])
    send.return_value = httpx.Response(200, headers=headers, stream=body)
    with pytest.raises(files.DownloadError, match="maximum size"):
        await files.download(PUBLIC_URL, max_bytes=5)
    assert body.reads == 2
    assert body.closed


@pytest.mark.parametrize(
    "status,headers", [(200, {"content-length": "100"}), (404, {})]
)
async def test_reject_without_reading_body(
    status: int, headers: dict[str, str], network: Any
) -> None:
    _, send = network
    body = Body([b"never read"])
    send.return_value = httpx.Response(status, headers=headers, stream=body)
    with pytest.raises(files.DownloadError):
        await files.download(PUBLIC_URL, max_bytes=5)
    assert body.reads == 0
    assert body.closed


async def test_decoded_size_limit(network: Any) -> None:
    _, send = network
    body = Body([gzip.compress(b"a" * 1000)])
    send.return_value = httpx.Response(
        200, headers={"content-encoding": "gzip"}, stream=body
    )
    with pytest.raises(files.DownloadError, match="maximum size"):
        await files.download(PUBLIC_URL, max_bytes=100)
    assert body.closed


async def test_exact_size_and_redirect_cleanup(network: Any) -> None:
    _, send = network
    redirect = Body([b"ignored"])
    final = Body([b"123", b"45"])
    send.side_effect = [
        httpx.Response(302, headers={"location": "/final"}, stream=redirect),
        httpx.Response(200, stream=final),
    ]
    assert await files.download(PUBLIC_URL, max_bytes=5) == (b"12345", None)
    assert redirect.closed and redirect.reads == 0
    assert final.closed


async def test_dns_failure_and_cancellation(network: Any) -> None:
    dns, send = network
    dns.side_effect = socket.gaierror("not found")
    with pytest.raises(files.DownloadError) as exc:
        await files.download(PUBLIC_URL)
    assert isinstance(exc.value.__cause__, socket.gaierror)
    dns.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await files.download(PUBLIC_URL)
    send.assert_not_called()


async def test_tcp_ip_and_tls_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real HTTP transport down to its TCP/TLS boundary."""
    # Proxy and CA environment settings must not replace the secure transport.
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("SSL_CERT_FILE", "/does/not/exist")
    stream = httpcore2.AsyncMockStream(
        [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"]
    )
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 8443))]
    with (
        mock.patch.object(
            asyncio.get_running_loop(),
            "getaddrinfo",
            side_effect=[records, AssertionError("second DNS lookup")],
        ) as dns,
        mock.patch.object(
            httpcore2.AnyIOBackend, "connect_tcp", return_value=stream
        ) as connect,
        mock.patch.object(stream, "start_tls", return_value=stream) as tls,
        mock.patch.object(stream, "write", wraps=stream.write) as write,
    ):
        assert await files.download(
            "https://media.example:8443/file?signature=a%2Fb"
        ) == (b"ok", None)
    dns.assert_called_once()
    assert connect.call_args.args[:2] == (PUBLIC_IP, 8443)
    assert tls.call_args.kwargs["server_hostname"] == "media.example"
    context = tls.call_args.kwargs["ssl_context"]
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    sent = b"".join(call.args[0] for call in write.call_args_list)
    assert b"Host: media.example:8443\r\n" in sent
    assert b"GET /file?signature=a%2Fb HTTP/1.1" in sent


@pytest.mark.parametrize("tls_failure", [False, True])
async def test_redirect_same_ip_has_independent_tls(tls_failure: bool) -> None:
    first = httpcore2.AsyncMockStream(
        [
            b"HTTP/1.1 302 Found\r\nLocation: https://other.example/final\r\n"
            b"Content-Length: 0\r\n\r\n"
        ]
    )
    second = httpcore2.AsyncMockStream(
        [b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"]
    )
    records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, 443))]
    with (
        mock.patch.object(
            asyncio.get_running_loop(), "getaddrinfo", return_value=records
        ),
        mock.patch.object(
            httpcore2.AnyIOBackend, "connect_tcp", side_effect=[first, second]
        ) as connect,
        mock.patch.object(first, "start_tls", return_value=first) as first_tls,
        mock.patch.object(
            second, "start_tls", return_value=second
        ) as second_tls,
        mock.patch.object(second, "write", wraps=second.write) as write,
    ):
        if tls_failure:
            second_tls.side_effect = httpcore2.ConnectError(
                "certificate mismatch"
            )
            with pytest.raises(files.DownloadError):
                await files.download(PUBLIC_URL)
            write.assert_not_called()
        else:
            assert await files.download(PUBLIC_URL) == (b"ok", None)
        assert connect.call_count == 2
        assert first_tls.call_args.kwargs["server_hostname"] == "media.example"
        assert second_tls.call_args.kwargs["server_hostname"] == "other.example"


@pytest.mark.parametrize(
    "target,cookie",
    [
        ("https://media.example.com/final", "access=ok"),
        ("https://cdn.example.com/final", "access=ok"),
        ("https://unrelated.test/final", None),
    ],
)
async def test_redirect_cookie_scope(
    target: str, cookie: str | None, network: Any
) -> None:
    _, send = network
    send.side_effect = [
        httpx.Response(
            302,
            headers={
                "location": target,
                "set-cookie": "access=ok; Domain=example.com; Path=/",
            },
        ),
        httpx.Response(200, content=b"ok"),
    ]
    assert await files.download("http://media.example.com/file") == (
        b"ok",
        None,
    )
    assert send.call_args.args[0].headers.get("cookie") == cookie


async def test_blocked_redirect_closes_body(network: Any) -> None:
    _, send = network
    body = Body([b"ignored"])
    send.return_value = httpx.Response(
        302, headers={"location": "http://127.0.0.1/"}, stream=body
    )
    with pytest.raises(files.DownloadError):
        await files.download(PUBLIC_URL)
    assert body.closed and body.reads == 0
    assert send.call_count == 1


async def test_dns_timeout(
    monkeypatch: pytest.MonkeyPatch, network: Any
) -> None:
    dns, send = network
    monkeypatch.setattr(files, "_DNS_TIMEOUT", 0)

    async def slow_dns(*args: Any, **kwargs: Any) -> None:
        await asyncio.Future()

    dns.side_effect = slow_dns
    with pytest.raises(files.DownloadError) as exc:
        await files.download(PUBLIC_URL)
    assert isinstance(exc.value.__cause__, TimeoutError)
    send.assert_not_called()


async def test_network_error_closes_body(network: Any) -> None:
    _, send = network

    class BrokenBody(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"123"
            raise httpx.ReadError("connection lost")

    body = BrokenBody([])
    send.return_value = httpx.Response(200, stream=body)
    with pytest.raises(files.DownloadError) as exc:
        await files.download(PUBLIC_URL)
    assert isinstance(exc.value.__cause__, httpx.ReadError)
    assert body.closed


async def test_report_loopback_repro() -> None:
    """A real listening internal service must receive no HTTP connections."""
    hits: list[bool] = []

    def connected(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        hits.append(True)
        writer.close()

    server = await asyncio.start_server(connected, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/internal"
        for media_type in ("image/png", "audio/wav", "application/pdf"):
            msg = messages.Message(
                role="user",
                parts=[messages.FilePart(data=url, media_type=media_type)],
            )
            with pytest.raises(files.DownloadError, match="Non-public IP"):
                await v3._messages_to_prompt([msg])
            if media_type != "image/png":
                with pytest.raises(files.DownloadError, match="Non-public IP"):
                    await openai_protocol._messages_to_openai([msg])
        with pytest.raises(files.DownloadError, match="Non-public IP"):
            await files.download(url)
        await asyncio.sleep(0)
        assert not hits
    # Closed and listening ports produce the same policy error, not port probes.
    with pytest.raises(files.DownloadError, match="Non-public IP"):
        await files.download(url)
    assert not hits


async def test_url_passthrough_and_inline_media_unchanged(network: Any) -> None:
    dns, send = network
    url = "https://media.example/image.png"
    part = messages.FilePart(data=url, media_type="image/png")
    msg = messages.Message(role="user", parts=[part])
    result = await v4._messages_to_prompt([msg])
    assert result[0]["content"][0]["data"] == {"type": "url", "url": url}
    result = await openai_protocol._messages_to_openai([msg])
    assert result[0]["content"][0]["image_url"]["url"] == url
    for data in (b"image", "data:image/png;base64,aW1hZ2U="):
        part = messages.FilePart(data=data, media_type="image/png")
        assert (await v3._file_part_to_wire(part))[
            "data"
        ] == "data:image/png;base64,aW1hZ2U="
    dns.assert_not_called()
    send.assert_not_called()
