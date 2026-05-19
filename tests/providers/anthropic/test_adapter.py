"""Tests for the Anthropic adapter's request shaping.

Focused on raw ``params`` passthrough and on the multi-turn round-trip
of provider-executed tool parts.
"""

from __future__ import annotations

from typing import Any, cast

import anthropic
import httpx
import pytest

import ai
from ai.providers.anthropic import protocol
from ai.types import messages

from .conftest import FakeAnthropicClient


class _RaisingMessages:
    def __init__(self, exc: anthropic.AnthropicError) -> None:
        self._exc = exc

    def stream(self, **kwargs: Any) -> Any:
        raise self._exc


class _RaisingAnthropicClient:
    def __init__(self, exc: anthropic.AnthropicError) -> None:
        self.messages = _RaisingMessages(exc)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[anthropic.AsyncAnthropic, dict[str, Any]]:
    _ = monkeypatch
    captured: dict[str, Any] = {}
    fake = FakeAnthropicClient(captured)
    return cast("anthropic.AsyncAnthropic", fake), captured


_MODEL = ai.Model("claude-sonnet-4-6", provider=ai.get_provider("anthropic"))


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def test_raw_params_pass_through_to_sdk_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captured = _patch_client(monkeypatch)

    await _drain(
        protocol.stream(
            fake,
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
            provider="anthropic",
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
    fake, _ = _patch_client(monkeypatch)

    stream = protocol.stream(
        fake,
        _MODEL,
        [ai.user_message("Hi")],
        params=[{"speed": "fast"}],
        provider="anthropic",
    )

    with pytest.raises(TypeError, match="dict"):
        await _drain(stream)


async def test_reasoning_signature_round_trips_from_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captured = _patch_client(monkeypatch)

    await _drain(
        protocol.stream(
            fake,
            _MODEL,
            [
                ai.assistant_message(
                    ai.thinking(
                        "hidden",
                        provider_metadata={
                            "provider": "anthropic",
                            "signature": "sig",
                        },
                    )
                ),
                ai.user_message("Hi"),
            ],
            provider="anthropic",
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
    """Built-in tool parts serialize back to wire."""
    fake, captured = _patch_client(monkeypatch)

    call = messages.BuiltinToolCallPart(
        tool_call_id="srvtoolu_1",
        tool_name="web_search",
        tool_args='{"query":"weather"}',
        provider_metadata={"provider": "anthropic"},
    )
    result = messages.BuiltinToolReturnPart(
        tool_call_id="srvtoolu_1",
        tool_name="web_search",
        result=[{"title": "Forecast", "url": "https://example.com"}],
        provider_metadata={
            "provider": "anthropic",
            "resultType": "web_search_tool_result",
        },
    )
    convo = [
        ai.user_message("What's the weather?"),
        messages.Message(role="assistant", parts=[call, result]),
        ai.user_message("Thanks"),
    ]

    await _drain(protocol.stream(fake, _MODEL, convo, provider="anthropic"))

    assistant = next(
        m for m in captured["messages"] if m["role"] == "assistant"
    )
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


async def test_sdk_errors_are_mapped_to_provider_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = monkeypatch
    response = httpx.Response(
        529,
        request=httpx.Request("POST", "https://anthropic.test/v1/messages"),
        headers={"request-id": "req-anthropic"},
    )
    sdk_error = anthropic.APIStatusError(
        "overloaded",
        response=response,
        body={"error": {"type": "overloaded_error"}},
    )
    fake = _RaisingAnthropicClient(sdk_error)

    with pytest.raises(ai.ProviderOverloadedError) as exc_info:
        await _drain(
            protocol.stream(
                cast("anthropic.AsyncAnthropic", fake),
                _MODEL,
                [ai.user_message("Hi")],
                provider="anthropic",
            )
        )

    exc = exc_info.value
    assert exc.provider == "anthropic"
    assert exc.http_context is not None
    assert exc.http_context.status_code == 529
    assert exc.http_context.request is response.request
    assert exc.http_context.response is response
    assert exc.request_id == "req-anthropic"
    assert exc.type == "overloaded_error"
    assert exc.__cause__ is sdk_error


async def test_model_404_is_mapped_to_model_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = monkeypatch
    response = httpx.Response(
        404,
        request=httpx.Request("POST", "https://anthropic.test/v1/messages"),
    )
    sdk_error = anthropic.NotFoundError(
        "model not found",
        response=response,
        body={"error": {"type": "not_found_error"}},
    )
    fake = _RaisingAnthropicClient(sdk_error)

    with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
        await _drain(
            protocol.stream(
                cast("anthropic.AsyncAnthropic", fake),
                _MODEL,
                [ai.user_message("Hi")],
                provider="anthropic",
            )
        )

    exc = exc_info.value
    assert isinstance(exc, ai.ProviderNotFoundError)
    assert exc.model_id == _MODEL.id
    assert exc.http_context is not None
    assert exc.http_context.status_code == 404
    assert exc.http_context.request is response.request
    assert exc.http_context.response is response
    assert exc.__cause__ is sdk_error
