"""Tiny test media built by hand: a solid-colour PNG and a one-line PDF."""

from __future__ import annotations

import struct
import zlib


def png(rgb: tuple[int, int, int] = (220, 20, 20), size: int = 64) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def pdf(text: str = "PINEAPPLE") -> bytes:
    stream = f"BT /F1 36 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return out


MCP_SERVER = r"""
import json, sys

def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    request = json.loads(line)
    method, rid = request.get("method"), request.get("id")
    if rid is None:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": request["params"].get(
                "protocolVersion", "2025-06-18"
            ),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "secret", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "secret_word",
            "description": "Returns today's secret word.",
            "inputSchema": {"type": "object", "properties": {}},
        }]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "MANGO"}]}
    else:
        error = {"code": -32601, "message": method}
        send({"jsonrpc": "2.0", "id": rid, "error": error})
        continue
    send({"jsonrpc": "2.0", "id": rid, "result": result})
"""
"""A dependency-free stdio MCP server with one tool, `secret_word`."""
