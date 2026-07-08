"""Terminal trace viewer: an OTLP/HTTP server that prints span trees.

Run::

    python -m ai.telemetry.utils.viewer [--port 4318]

and point any OTLP/HTTP exporter at ``http://127.0.0.1:4318/v1/traces``::

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter("http://127.0.0.1:4318/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    ai.telemetry.otel.install()

Spans are buffered per trace, and the tree is printed when the trace's
root span arrives (roots end — and therefore export — last).
"""

from __future__ import annotations

import argparse
import sys
from http import server
from typing import Any

from ... import errors

try:
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
except ModuleNotFoundError as exc:  # pragma: no cover
    raise errors.InstallationError(
        "could not import `opentelemetry.proto`, which the telemetry viewer "
        "needs to decode OTLP, you can install it with "
        "`pip install opentelemetry-exporter-otlp-proto-http`"
    ) from exc

_traces: dict[bytes, list[Any]] = {}


def _value(any_value: Any) -> Any:
    kind = any_value.WhichOneof("value")
    return getattr(any_value, kind) if kind else None


def _line(span: Any) -> str:
    attrs = {kv.key: _value(kv.value) for kv in span.attributes}
    duration = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9
    tokens = ""
    if "gen_ai.usage.input_tokens" in attrs:
        tokens = (
            f"  in:{attrs['gen_ai.usage.input_tokens']}"
            f" out:{attrs.get('gen_ai.usage.output_tokens', '?')} tok"
        )
    replay = "↻ " if attrs.get("ai.replay") else ""
    error = "  ✗" if span.status.code == 2 else ""  # STATUS_CODE_ERROR
    return f"{replay}{span.name}{tokens}  {duration:.2f}s{error}"


def _print_trace(spans: list[Any]) -> None:
    children: dict[bytes, list[Any]] = {}
    roots = []
    for span in spans:
        if span.parent_span_id:
            children.setdefault(span.parent_span_id, []).append(span)
        else:
            roots.append(span)

    lines = [f"trace {spans[0].trace_id.hex()}"]

    def render(span: Any, prefix: str, kid_prefix: str) -> None:
        lines.append(prefix + _line(span))
        kids = sorted(
            children.get(span.span_id, []),
            key=lambda s: s.start_time_unix_nano,
        )
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1
            render(
                kid,
                kid_prefix + ("└─ " if last else "├─ "),
                kid_prefix + ("   " if last else "│  "),
            )

    for root in roots:
        render(root, "", "")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


class _Handler(server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — stdlib API name
        length = int(self.headers.get("content-length") or 0)
        request = trace_service_pb2.ExportTraceServiceRequest()
        request.ParseFromString(self.rfile.read(length))
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    _traces.setdefault(span.trace_id, []).append(span)
                    if not span.parent_span_id:
                        _print_trace(_traces.pop(span.trace_id))
        payload = trace_service_pb2.ExportTraceServiceResponse()
        body = payload.SerializeToString()
        self.send_response(200)
        self.send_header("content-type", "application/x-protobuf")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print OTLP traces to the terminal as trees."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4318)
    args = parser.parse_args()
    httpd = server.HTTPServer((args.host, args.port), _Handler)
    sys.stdout.write(f"listening on http://{args.host}:{args.port}/v1/traces\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
