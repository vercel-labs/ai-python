"""Tests for the OpenAI adapter's request shaping.

Focused on non-trivial mappings between :class:`OpenAIChatParams` and
the SDK kwargs (translations, derived fields) and on the explicit
guards against unsupported built-in tool surfaces. Pure passthroughs
are not asserted here — those are pydantic / typechecker territory.
"""

from __future__ import annotations

from typing import Any

import pydantic
import pytest

import ai
from ai import models
from ai.models.openai import adapter, openai
from ai.models.openai import params as openai_params
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


async def test_system_message_mode_developer_rewrites_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.system_message("rules"), ai.user_message("Hi")],
            params=openai_params.OpenAIChatParams(system_message_mode="developer"),
        )
    )

    assert captured["messages"][0] == {"role": "developer", "content": "rules"}


async def test_logprobs_int_translates_to_top_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``logprobs=N`` becomes ``logprobs=True`` + ``top_logprobs=N``."""
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=openai_params.OpenAIChatParams(logprobs=3),
        )
    )

    assert captured["logprobs"] is True
    assert captured["top_logprobs"] == 3


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
            params=openai_params.OpenAIChatParams(strict_json_schema=False),
        )
    )

    assert captured["response_format"]["json_schema"]["strict"] is False


async def test_text_verbosity_aliased_to_verbosity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=openai_params.OpenAIChatParams(text_verbosity="low"),
        )
    )

    assert captured["verbosity"] == "low"


async def test_extra_body_and_headers_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch(monkeypatch)

    await _drain(
        adapter.stream(
            _TEST_CLIENT,
            _MODEL,
            [ai.user_message("Hi")],
            params=openai_params.OpenAIChatParams(
                extra_body={"future_option": True},
                extra_headers={"x-openai-feature": "enabled"},
            ),
        )
    )

    assert captured["extra_body"] == {"future_option": True}
    assert captured["extra_headers"] == {"x-openai-feature": "enabled"}


async def test_invalid_params_rejected_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch)

    stream = adapter.stream(
        _TEST_CLIENT,
        _MODEL,
        [ai.user_message("Hi")],
        params={"reasoning_effort": "high"},
    )

    with pytest.raises(TypeError, match="OpenAIChatParams"):
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
