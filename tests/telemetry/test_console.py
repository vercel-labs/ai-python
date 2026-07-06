"""Console adapter: live lines and end-of-trace tree."""

from __future__ import annotations

import io

import ai
from ai.telemetry import console


async def test_console_prints_tree() -> None:
    out = io.StringIO()
    adapter = console.ConsoleAdapter(out=out)
    ai.telemetry.register(adapter)
    try:
        async with ai.telemetry.span("outer"):
            async with ai.telemetry.span("inner", k=1):
                pass
    finally:
        ai.telemetry.unregister(adapter)

    text = out.getvalue()
    assert "▸ outer" in text
    assert "▸   inner (k=1)" in text
    assert "└─ inner (k=1)" in text
    assert "trace " in text
