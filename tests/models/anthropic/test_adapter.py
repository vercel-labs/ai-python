"""Tests for the Anthropic adapter's request shaping.

Focused on non-trivial mappings between :class:`AnthropicParams` and
the SDK kwargs (synthesized objects, header merging, conditional
omission) and on the multi-turn round-trip of provider-executed tool
parts. Pure passthroughs are not asserted here — those are pydantic /
typechecker territory.
"""

from __future__ import annotations

from typing import Any

import pytest

import ai
from ai import models
from ai.models.anthropic import adapter, anthropic
from ai.models.anthropic import metadata as anthropic_metadata
from ai.models.anthropic import params as anthropic_params
from ai.types import messages

from .conftest import FakeAnthropicClient


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeAnthropicClient, dict[str, Any]]:
    captured: dict[str, Any] = {}
    fake = FakeAnthropicClient(captured)
    monkeypatch.setattr(adapter, "_make_client", lambda client: fake)
    return fake, captured


_TEST_CLIENT = models.Client(base_url="https://anthropic.test", api_key="sk-test")
_MODEL = anthropic("claude-sonnet-4-6")


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def test_output_config_combines_effort_and_task_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``effort`` + ``task_budget`` are squashed into one ``output_config`` block."""
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=anthropic_params.AnthropicParams(
                effort="high",
                task_budget=anthropic_params.AnthropicTaskBudget(
                    type="tokens", total=20000
                ),
            ),
        )
    )

    assert captured["output_config"] == {
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 20000},
    }


async def test_disable_parallel_tool_use_synthesizes_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=anthropic_params.AnthropicParams(disable_parallel_tool_use=True),
        )
    )

    assert captured["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


async def test_extra_headers_override_merged_anthropic_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-supplied ``anthropic-beta`` wins over adapter-generated values."""
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=anthropic_params.AnthropicParams(
                betas=["adapter-beta-2026-01-01"],
                extra_headers={
                    "anthropic-beta": "user-override-2026-01-01",
                    "x-anthropic-feature": "enabled",
                },
            ),
        )
    )

    assert captured["extra_headers"] == {
        "anthropic-beta": "user-override-2026-01-01",
        "x-anthropic-feature": "enabled",
    }


async def test_extra_body_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=anthropic_params.AnthropicParams(
                extra_body={"future_option": {"enabled": True}},
            ),
        )
    )

    assert captured["extra_body"] == {"future_option": {"enabled": True}}


async def test_invalid_params_rejected_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)

    stream = adapter.stream(
        _TEST_CLIENT,
        _MODEL,
        [ai.user_message("Hi")],
        params={"speed": "fast"},
    )

    with pytest.raises(TypeError, match="AnthropicParams"):
        await _drain(stream)


async def test_disabled_thinking_is_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AnthropicDisabledThinking`` must be omitted from the SDK kwargs."""
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=anthropic_params.AnthropicParams(
                thinking=anthropic_params.AnthropicDisabledThinking(type="disabled"),
            ),
        )
    )

    assert "thinking" not in captured


async def test_send_reasoning_false_strips_thinking_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``send_reasoning=False`` must not emit ``thinking`` blocks in messages."""
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [
                ai.assistant_message(
                    ai.thinking(
                        "hidden",
                        provider_metadata=anthropic_metadata.AnthropicProviderMetadata(
                            signature="sig"
                        ),
                    )
                ),
                ai.user_message("Hi"),
            ],
            params=anthropic_params.AnthropicParams(send_reasoning=False),
        )
    )

    # The assistant message had only a reasoning part; with reasoning
    # stripped it produces no content and is dropped.
    assert captured["messages"] == [{"role": "user", "content": "Hi"}]


async def test_reasoning_signature_round_trips_from_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [
                ai.assistant_message(
                    ai.thinking(
                        "hidden",
                        provider_metadata=anthropic_metadata.AnthropicProviderMetadata(
                            signature="sig"
                        ),
                    )
                ),
                ai.user_message("Hi"),
            ],
        )
    )

    assert captured["messages"][0] == {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "hidden",
                "signature": "sig",
            }
        ],
    }


async def test_builtin_tool_parts_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BuiltinToolCallPart``/``BuiltinToolReturnPart`` serialize back to wire."""
    _, captured = _patch_client(monkeypatch)

    call = messages.BuiltinToolCallPart(
        tool_call_id="srvtoolu_1",
        tool_name="web_search",
        tool_args='{"query":"weather"}',
        provider_metadata=anthropic_metadata.AnthropicProviderMetadata(),
    )
    result = messages.BuiltinToolReturnPart(
        tool_call_id="srvtoolu_1",
        tool_name="web_search",
        result=[{"title": "Forecast", "url": "https://example.com"}],
        provider_metadata=anthropic_metadata.AnthropicProviderMetadata(
            result_type="web_search_tool_result"
        ),
    )
    convo = [
        ai.user_message("What's the weather?"),
        messages.Message(role="assistant", parts=[call, result]),
        ai.user_message("Thanks"),
    ]

    await _drain(adapter.stream(_TEST_CLIENT, _MODEL, convo))

    assistant = next(m for m in captured["messages"] if m["role"] == "assistant")
    assert assistant["content"] == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "weather"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{"title": "Forecast", "url": "https://example.com"}],
        },
    ]
