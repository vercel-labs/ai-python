"""Tests for the Anthropic adapter's request shaping.

Focused on raw ``params`` passthrough and on the multi-turn round-trip
of provider-executed tool parts.
"""

from __future__ import annotations

from typing import Any

import pytest

import ai
from ai import models
from ai.models.anthropic import adapter, anthropic
from ai.models.anthropic import metadata as anthropic_metadata
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


async def test_raw_params_pass_through_to_sdk_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, captured = _patch_client(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params={
                "max_tokens": 123,
                "speed": "fast",
                "thinking": {"type": "disabled"},
                "output_config": {
                    "effort": "high",
                    "task_budget": {"type": "tokens", "total": 20000},
                },
                "tool_choice": {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
                },
                "extra_body": {"future_option": {"enabled": True}},
                "extra_headers": {"x-anthropic-feature": "enabled"},
            },
        )
    )

    assert captured["max_tokens"] == 123
    assert captured["speed"] == "fast"
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["output_config"] == {
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 20000},
    }
    assert captured["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert captured["extra_body"] == {"future_option": {"enabled": True}}
    assert captured["extra_headers"] == {"x-anthropic-feature": "enabled"}


async def test_non_dict_params_rejected_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)

    stream = adapter.stream(
        _TEST_CLIENT,
        _MODEL,
        [ai.user_message("Hi")],
        params=[{"speed": "fast"}],
    )

    with pytest.raises(TypeError, match="dict"):
        await _drain(stream)


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
