"""Console adapter: prints spans to a terminal as they happen.

Experimental: not part of the stable API, may change or be removed.

Register it like any adapter::

    from ai.experimental_telemetry.utils import console
    ai.experimental_telemetry.register(console.ConsoleAdapter())

Prints one ``▸`` line when a span starts (so long runs are visible
live) and, when a trace's root span ends, the whole tree with
durations.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ... import experimental_telemetry as telemetry

if TYPE_CHECKING:
    from typing import TextIO


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _label(sp: telemetry.Span) -> str:
    match sp.data:
        case telemetry.AiStreamSpanData() as d:
            tokens = (
                f"  in:{d.usage.input_tokens} out:{d.usage.output_tokens} tok"
                if d.usage is not None
                else ""
            )
            return f"chat {d.model}{tokens}"
        case telemetry.AiGenerateSpanData() as d:
            return f"generate {d.model}"
        case telemetry.ToolExecutionSpanData() as d:
            args = ", ".join(f"{k}={v!r}" for k, v in (d.args or {}).items())
            return f"tool {d.tool_name}({_short(args)})"
        case telemetry.HookSpanData() as d:
            return f"hook {d.label} {d.hook_type} [{d.status}]"
        case telemetry.RunSpanData() as d:
            return f"run {d.agent} ({d.model})"
        case telemetry.CustomSpanData() as d:
            attrs = ", ".join(f"{k}={v!r}" for k, v in d.attributes.items())
            return sp.name + (f" ({_short(attrs)})" if attrs else "")
        case _:
            return sp.name


def _line(sp: telemetry.Span) -> str:
    end = sp.ended_at or sp.started_at
    # A span's lifetime can extend past the response (tool dispatch
    # while the stream is open); when the milestone is there, report
    # the true response latency instead.
    for ev in sp.span_events:
        if ev.name == telemetry.RESPONSE_COMPLETE:
            end = ev.time_ns
            break
    duration = (end - sp.started_at) / 1e9
    replay = "↻ " if sp.replay else ""
    error = (
        f"  ✗ {type(sp.error).__name__}: {_short(str(sp.error))}"
        if sp.error is not None
        else ""
    )
    return f"{replay}{_label(sp)}  {duration:.2f}s{error}"


class ConsoleAdapter:
    """Print spans to ``out`` (default: stdout)."""

    def __init__(self, *, out: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout
        self._depth: dict[str, int] = {}
        self._ended: dict[str, list[telemetry.Span]] = {}

    def on_span_start(self, span: telemetry.Span) -> None:
        depth = 0
        if span.parent_id is not None:
            depth = self._depth.get(span.parent_id, -1) + 1
        self._depth[span.id] = depth
        replay = "↻ " if span.replay else ""
        self._out.write(f"▸ {'  ' * depth}{replay}{_label(span)}\n")

    def on_span_event(
        self, span: telemetry.Span, event: telemetry.SpanEvent
    ) -> None:
        depth = self._depth.get(span.id, 0) + 1
        offset_ms = (event.time_ns - span.started_at) / 1e6
        attrs = ", ".join(f"{k}={v!r}" for k, v in event.attributes.items())
        suffix = f" ({_short(attrs)})" if attrs else ""
        self._out.write(
            f"· {'  ' * depth}{event.name} +{offset_ms:.0f}ms{suffix}\n"
        )

    def on_span_end(self, span: telemetry.Span) -> None:
        self._ended.setdefault(span.trace_id, []).append(span)
        if span.parent_id is not None:
            return

        # Root ended: print the tree and forget the trace.
        spans = self._ended.pop(span.trace_id)
        for s in spans:
            self._depth.pop(s.id, None)
        children: dict[str | None, list[telemetry.Span]] = {}
        for s in spans:
            children.setdefault(s.parent_id, []).append(s)

        lines = [f"trace {span.trace_id}"]

        def render(s: telemetry.Span, prefix: str, kid_prefix: str) -> None:
            lines.append(prefix + _line(s))
            kids = sorted(children.get(s.id, []), key=lambda c: c.started_at)
            for i, kid in enumerate(kids):
                last = i == len(kids) - 1
                render(
                    kid,
                    kid_prefix + ("└─ " if last else "├─ "),
                    kid_prefix + ("   " if last else "│  "),
                )

        render(span, "", "")
        self._out.write("\n".join(lines) + "\n")
