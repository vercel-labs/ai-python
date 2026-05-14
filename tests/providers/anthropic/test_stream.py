"""Tests for Anthropic stream parsing of provider-executed tools.

The adapter consumes anthropic SDK ``content_block_*`` events and
emits :class:`ai.types.events.BuiltinToolStart`/``Delta``/``End`` for
``server_tool_use`` blocks and :class:`BuiltinToolResult` for
``*_tool_result`` blocks. Drained through :class:`models.Stream` to
also exercise event aggregation in ``core.api``.
"""

from __future__ import annotations

import pytest

import ai
from ai import models
from ai.providers.anthropic import adapter
from ai.types import events, messages

from .conftest import (
    FakeAnthropicClient,
    FakeStream,
    block_delta,
    block_start,
    block_stop,
    snapshot_block,
)

_MODEL = ai.Model("claude-sonnet-4-6", provider=ai.get_provider("anthropic"))


async def _drain(stream: FakeStream, monkeypatch: pytest.MonkeyPatch) -> models.Stream:
    fake = FakeAnthropicClient(stream=stream)
    monkeypatch.setattr(adapter, "_make_client", lambda model: fake)
    s = models.Stream(adapter.stream(_MODEL, [ai.user_message("Hi")]))
    async for _ in s:
        pass
    return s


async def test_server_tool_use_emits_builtin_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_events = [
        block_start(0, "server_tool_use", id="srvtoolu_1", name="web_search"),
        block_delta(0, "input_json_delta", partial_json='{"query"'),
        block_delta(0, "input_json_delta", partial_json=':"weather"}'),
        block_stop(0),
    ]
    s = await _drain(FakeStream(sdk_events), monkeypatch)

    calls = s.message.builtin_tool_calls
    assert len(calls) == 1
    assert calls[0].tool_call_id == "srvtoolu_1"
    assert calls[0].tool_name == "web_search"
    assert calls[0].tool_args == '{"query":"weather"}'
    assert calls[0].provider_metadata == {"provider": "anthropic"}


async def test_tool_result_block_emits_builtin_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result blocks are emitted on ``content_block_stop`` after lookup
    in the message snapshot for the matching ``server_tool_use``'s name."""
    payload = [{"title": "Forecast", "url": "https://example.com"}]
    snapshot = [
        snapshot_block("server_tool_use", id="srvtoolu_1", name="web_search"),
        snapshot_block(
            "web_search_tool_result",
            tool_use_id="srvtoolu_1",
            content=payload,
        ),
    ]
    sdk_events = [
        # The server_tool_use lifecycle (start + stop) so the adapter
        # registers the block type at index 0.
        block_start(0, "server_tool_use", id="srvtoolu_1", name="web_search"),
        block_stop(0),
        # Then the result block at index 1.
        block_start(1, "web_search_tool_result"),
        block_stop(1),
    ]
    s = await _drain(FakeStream(sdk_events, snapshot_content=snapshot), monkeypatch)

    returns = s.message.builtin_tool_returns
    assert len(returns) == 1
    ret = returns[0]
    assert ret.tool_call_id == "srvtoolu_1"
    assert ret.tool_name == "web_search"
    assert ret.result == payload
    assert ret.provider_metadata == {
        "provider": "anthropic",
        "resultType": "web_search_tool_result",
    }


async def test_signature_delta_emits_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_events = [
        block_start(0, "thinking"),
        block_delta(0, "thinking_delta", thinking="hidden"),
        block_delta(0, "signature_delta", signature="sig"),
        block_stop(0),
    ]
    s = await _drain(FakeStream(sdk_events), monkeypatch)

    reasoning = s.message.parts[0]
    assert isinstance(reasoning, messages.ReasoningPart)
    assert reasoning.provider_metadata == {
        "provider": "anthropic",
        "signature": "sig",
    }


async def test_event_kinds_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Builtin tool events come through the adapter in the right order."""
    sdk_events = [
        block_start(0, "server_tool_use", id="srvtoolu_1", name="web_search"),
        block_delta(0, "input_json_delta", partial_json="{}"),
        block_stop(0),
    ]
    fake = FakeAnthropicClient(stream=FakeStream(sdk_events))
    monkeypatch.setattr(adapter, "_make_client", lambda model: fake)

    seen: list[type] = []
    async for event in adapter.stream(_MODEL, [ai.user_message("Hi")]):
        seen.append(type(event))

    assert seen == [
        events.StreamStart,
        events.BuiltinToolStart,
        events.BuiltinToolDelta,
        events.BuiltinToolEnd,
        events.StreamEnd,
    ]


async def test_builtin_tool_end_carries_call_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BuiltinToolEnd.tool_call`` exposes the aggregated part from Stream."""
    sdk_events = [
        block_start(0, "server_tool_use", id="srvtoolu_42", name="web_search"),
        block_delta(0, "input_json_delta", partial_json='{"q":"x"}'),
        block_stop(0),
    ]
    fake = FakeAnthropicClient(stream=FakeStream(sdk_events))
    monkeypatch.setattr(adapter, "_make_client", lambda model: fake)

    end_event: events.BuiltinToolEnd | None = None
    s = models.Stream(adapter.stream(_MODEL, [ai.user_message("Hi")]))
    async for event in s:
        if isinstance(event, events.BuiltinToolEnd):
            end_event = event

    assert end_event is not None
    assert isinstance(end_event.tool_call, messages.BuiltinToolCallPart)
    assert end_event.tool_call.tool_call_id == "srvtoolu_42"
    assert end_event.tool_call.tool_args == '{"q":"x"}'
