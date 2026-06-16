"""Tests for the Anthropic built-in tool surface.

The adapter must:

1. Translate provider ``Tool`` declarations to their registered wire
   ``type`` / ``name`` and field set.
2. Add adapter-owned provider-tool betas unless the caller supplied an
   ``anthropic-beta`` request header.
3. Reject mutually-exclusive domain filters at construction time
   (``allowed_domains`` + ``blocked_domains``).
"""

from __future__ import annotations

from typing import Any, cast

import anthropic
import pytest

import ai
from ai.providers.anthropic import protocol
from ai.providers.anthropic import tools as anthropic_tools

from .conftest import FakeAnthropicClient

_MODEL = ai.get_model("anthropic:claude-sonnet-4-6")


async def _capture_tools(
    monkeypatch: pytest.MonkeyPatch,
    tools: list[Any],
    *,
    params: ai.InferenceRequestParams | None = None,
) -> dict[str, Any]:
    _ = monkeypatch
    captured: dict[str, Any] = {}
    fake = cast("anthropic.AsyncAnthropic", FakeAnthropicClient(captured))
    stream = protocol.stream(
        fake,
        _MODEL,
        [ai.user_message("Hi")],
        tools=tools,
        params=params,
        provider="anthropic",
    )
    async for _ in stream:
        pass
    return captured


def _beta_header(captured: dict[str, Any]) -> str | None:
    return (captured.get("extra_headers") or {}).get("anthropic-beta")


async def test_web_search_full_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.web_search(
                max_uses=3,
                allowed_domains=["example.com"],
                user_location=anthropic_tools.UserLocation(
                    city="SF",
                    country="US",
                ),
            )
        ],
    )

    assert captured["tools"] == [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 3,
            "allowed_domains": ["example.com"],
            "user_location": {
                "type": "approximate",
                "city": "SF",
                "country": "US",
            },
        }
    ]
    assert _beta_header(captured) == "code-execution-web-tools-2026-02-09"


async def test_web_fetch_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.web_fetch(
                citations=anthropic_tools.Citations(enabled=True),
            )
        ],
    )

    assert captured["tools"] == [
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "citations": {"enabled": True},
        }
    ]


async def test_computer_use_required_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.computer_use(
                display_width_px=1024,
                display_height_px=768,
            )
        ],
    )

    assert captured["tools"] == [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": 1024,
            "display_height_px": 768,
        }
    ]
    assert _beta_header(captured) == "computer-use-2025-11-24"


async def test_text_editor_name_differs_from_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TextEditor`` ships under the ``str_replace_based_edit_tool`` name."""
    captured = await _capture_tools(
        monkeypatch, [anthropic_tools.text_editor()]
    )

    assert captured["tools"] == [
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
    ]


async def test_bare_tools_emit_only_type_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.code_execution(),
            anthropic_tools.bash(),
            anthropic_tools.memory(),
        ],
    )

    assert captured["tools"] == [
        {"type": "code_execution_20260120", "name": "code_execution"},
        {"type": "bash_20250124", "name": "bash"},
        {"type": "memory_20250818", "name": "memory"},
    ]


async def test_user_anthropic_beta_header_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw ``extra_headers`` pass through instead of being rewritten."""
    captured = await _capture_tools(
        monkeypatch,
        [
            anthropic_tools.web_search(),
            anthropic_tools.web_fetch(),
        ],
        params=ai.InferenceRequestParams(
            extra_headers={"anthropic-beta": "custom-beta-2026-01-01"}
        ),
    )

    assert _beta_header(captured) == "custom-beta-2026-01-01"


def test_web_search_rejects_mutually_exclusive_domains() -> None:
    with pytest.raises(ValueError, match="only one of"):
        anthropic_tools.web_search(
            allowed_domains=["a.example"],
            blocked_domains=["b.example"],
        )


def test_web_fetch_rejects_mutually_exclusive_domains() -> None:
    with pytest.raises(ValueError, match="only one of"):
        anthropic_tools.web_fetch(
            allowed_domains=["a.example"],
            blocked_domains=["b.example"],
        )
