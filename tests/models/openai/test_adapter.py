"""Tests for the OpenAI adapter's request shaping.

Focused on raw ``params`` passthrough, adapter-owned structured-output
formatting, and explicit guards against unsupported built-in tool surfaces.
"""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

import ai
from ai import models
from ai.models.openai import adapter, openai
from ai.models.openai import tools as openai_tools
from ai.types import messages


class _Answer(pydantic.BaseModel):
    answer: str


class _EmptyOpenAIStream:
    def __aiter__(self) -> _EmptyOpenAIStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _FakeCompletions:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    async def create(self, **kwargs: Any) -> _EmptyOpenAIStream:
        self._captured.update(kwargs)
        return _EmptyOpenAIStream()


class _FakeChat:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.completions = _FakeCompletions(captured)


class _FakeOpenAIClient:
    def __init__(self, captured: dict[str, Any]) -> None:
        self.chat = _FakeChat(captured)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


_TEST_CLIENT = models.Client(base_url="https://openai.test", api_key="sk-test")
_MODEL = openai("gpt-5.4")


def _patch(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    fake = _FakeOpenAIClient(captured)
    monkeypatch.setattr(adapter, "_make_client", lambda client: fake)
    return captured


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def test_system_messages_use_openai_system_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.system_message("rules"), ai.user_message("Hi")],
        )
    )

    assert captured["messages"][0] == {"role": "system", "content": "rules"}


async def test_raw_params_pass_through_to_sdk_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params={
                "logprobs": 3,
                "verbosity": "low",
                "max_completion_tokens": 128,
                "extra_body": {"future_option": True},
                "extra_headers": {"x-openai-feature": "enabled"},
                "stream_options": {"include_usage": False, "custom": True},
            },
        )
    )

    assert captured["logprobs"] == 3
    assert captured["verbosity"] == "low"
    assert captured["max_completion_tokens"] == 128
    assert captured["extra_body"] == {"future_option": True}
    assert captured["extra_headers"] == {"x-openai-feature": "enabled"}
    assert captured["stream_options"] == {"include_usage": False, "custom": True}


async def test_strict_json_schema_flows_into_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            output_type=_Answer,
        )
    )

    assert captured["response_format"]["json_schema"]["strict"] is True


async def test_non_dict_params_rejected_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)

    stream = adapter.stream(
        _TEST_CLIENT,
        _MODEL,
        [ai.user_message("Hi")],
        params=[{"reasoning_effort": "high"}],
    )

    with pytest.raises(TypeError, match="dict"):
        await _drain(stream)


async def test_builtin_tool_in_request_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat-completions adapter rejects OpenAI built-in tools at the boundary."""
    _patch(monkeypatch)

    stream = adapter.stream(
        _TEST_CLIENT,
        _MODEL,
        [ai.user_message("Hi")],
        tools=[openai_tools.web_search()],
    )

    with pytest.raises(NotImplementedError, match="Responses API"):
        await _drain(stream)


async def test_builtin_part_in_messages_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BuiltinToolCallPart`` cannot round-trip through chat-completions."""
    _patch(monkeypatch)

    convo = [
        ai.user_message("Hi"),
        messages.Message(
            role="assistant",
            parts=[
                messages.BuiltinToolCallPart(
                    tool_call_id="srvtoolu_1",
                    tool_name="web_search",
                ),
            ],
        ),
    ]

    with pytest.raises(NotImplementedError, match="BuiltinTool"):
        await _drain(adapter.stream(_TEST_CLIENT, _MODEL, convo))
