"""Tests for the OpenAI adapter's request shaping.

Focused on raw ``params`` passthrough, adapter-owned structured-output
formatting, and explicit guards against unsupported built-in tool surfaces.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import openai
import pydantic
import pytest

import ai
from ai.providers.openai import protocol
from ai.providers.openai import tools as openai_tools
from ai.types import events, messages, tools


class _Answer(pydantic.BaseModel):
    answer: str


class _EmptyOpenAIStream:
    def __aiter__(self) -> _EmptyOpenAIStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _ListStream:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self._idx = 0

    def __aiter__(self) -> _ListStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


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


class _FakeResponses:
    def __init__(
        self, captured: dict[str, Any], items: list[dict[str, Any]]
    ) -> None:
        self._captured = captured
        self._items = items

    async def create(self, **kwargs: Any) -> _ListStream:
        self._captured.update(kwargs)
        return _ListStream(self._items)


class _FakeResponsesClient:
    def __init__(
        self, captured: dict[str, Any], items: list[dict[str, Any]]
    ) -> None:
        self.responses = _FakeResponses(captured, items)


class _RaisingCompletions:
    def __init__(self, exc: openai.OpenAIError) -> None:
        self._exc = exc

    async def create(self, **kwargs: Any) -> _EmptyOpenAIStream:
        raise self._exc


class _RaisingChat:
    def __init__(self, exc: openai.OpenAIError) -> None:
        self.completions = _RaisingCompletions(exc)


class _RaisingOpenAIClient:
    def __init__(self, exc: openai.OpenAIError) -> None:
        self.chat = _RaisingChat(exc)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


_MODEL = ai.Model(
    "gpt-5.4",
    provider=ai.get_provider("openai", api_key="sk-test"),
)


def _patch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[openai.AsyncOpenAI, dict[str, Any]]:
    _ = monkeypatch
    captured: dict[str, Any] = {}
    fake = _FakeOpenAIClient(captured)
    return cast("openai.AsyncOpenAI", fake), captured


def _patch_responses(
    items: list[dict[str, Any]] | None = None,
) -> tuple[openai.AsyncOpenAI, dict[str, Any]]:
    captured: dict[str, Any] = {}
    fake = _FakeResponsesClient(captured, items or [])
    return cast("openai.AsyncOpenAI", fake), captured


async def _drain(stream: Any) -> None:
    async for _ in stream:
        pass


async def test_responses_request_uses_responses_input() -> None:
    fake, captured = _patch_responses()

    await _drain(
        protocol.OpenAIResponsesProtocol().stream(
            fake,
            _MODEL,
            [ai.system_message("rules"), ai.user_message("Hi")],
            provider="openai",
        )
    )

    assert captured["model"] == "gpt-5.4"
    assert captured["stream"] is True
    assert captured["input"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
    ]
    assert "messages" not in captured


async def test_responses_params_and_structured_output() -> None:
    fake, captured = _patch_responses()

    await _drain(
        protocol.OpenAIResponsesProtocol().stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            output_type=_Answer,
            params=ai.InferenceRequestParams(
                reasoning=ai.ReasoningParams(effort="high"),
                context_management=ai.ContextManagementParams(
                    compaction=ai.TokenThreshold(120_000)
                ),
                output=ai.OutputParams(
                    include=frozenset({"file_search_call.results"}),
                    reasoning_summary="auto",
                    text_verbosity="low",
                ),
                extra_body={"future_option": True},
                extra_headers={"x-openai-feature": "enabled"},
            ),
            provider="openai",
        )
    )

    assert captured["reasoning"] == {"effort": "high", "summary": "auto"}
    assert captured["context_management"] == [
        {"type": "compaction", "compact_threshold": 120_000}
    ]
    assert captured["include"] == ["file_search_call.results"]
    assert captured["extra_headers"] == {"x-openai-feature": "enabled"}
    assert captured["extra_body"] == {"future_option": True}
    assert captured["text"]["verbosity"] == "low"
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["name"] == "_Answer"
    assert captured["text"]["format"]["strict"] is True


async def test_chat_rejects_text_verbosity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = _patch(monkeypatch)

    with pytest.raises(ValueError, match="text verbosity"):
        await _drain(
            protocol.stream(
                fake,
                _MODEL,
                [ai.user_message("Hi")],
                params=ai.InferenceRequestParams(
                    output=ai.OutputParams(text_verbosity="low")
                ),
                provider="openai",
            )
        )


async def test_chat_rejects_context_management(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = _patch(monkeypatch)

    with pytest.raises(ValueError, match="context management"):
        await _drain(
            protocol.stream(
                fake,
                _MODEL,
                [ai.user_message("Hi")],
                params=ai.InferenceRequestParams(
                    context_management=ai.ContextManagementParams(
                        compaction=ai.TokenThreshold(120_000)
                    )
                ),
                provider="openai",
            )
        )


async def test_responses_tools_convert_function_and_provider_tools() -> None:
    fake, captured = _patch_responses()

    await _drain(
        protocol.OpenAIResponsesProtocol().stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            tools=[
                tools.Tool(
                    kind="function",
                    name="weather",
                    args=tools.FunctionToolArgs(
                        description="Get weather",
                        params={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    ),
                ),
                openai_tools.web_search(search_context_size="low"),
                openai_tools.code_interpreter(),
            ],
            provider="openai",
        )
    )

    assert captured["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
        {"type": "web_search", "search_context_size": "low"},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ]


async def test_responses_streams_text_and_usage() -> None:
    fake, _ = _patch_responses(
        [
            {
                "type": "response.created",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": "msg_1", "type": "message", "role": "assistant"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "Hi",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "model": "gpt-5.4",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 3,
                        "input_tokens_details": {"cached_tokens": 1},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            },
        ]
    )

    stream = ai.Stream(
        protocol.OpenAIResponsesProtocol().stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            provider="openai",
        )
    )
    async for _ in stream:
        pass

    assert stream.text == "Hi"
    assert stream.usage is not None
    assert stream.usage.input_tokens == 3
    assert stream.usage.output_tokens == 5
    assert stream.usage.reasoning_tokens == 2
    assert stream.usage.cache_read_tokens == 1
    assert stream.message.provider_metadata == {
        "openai": {
            "response_id": "resp_1",
            "model": "gpt-5.4",
            "status": "completed",
        }
    }


async def test_responses_streams_function_tool_call() -> None:
    fake, _ = _patch_responses(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": '{"city"',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": ':"SF"}',
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": '{"city":"SF"}',
                },
            },
        ]
    )

    stream = ai.Stream(
        protocol.OpenAIResponsesProtocol().stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            provider="openai",
        )
    )
    async for _ in stream:
        pass

    assert len(stream.tool_calls) == 1
    assert stream.tool_calls[0].tool_call_id == "call_1"
    assert stream.tool_calls[0].tool_name == "weather"
    assert stream.tool_calls[0].tool_args == '{"city":"SF"}'


async def test_responses_streams_builtin_tool_call_and_result() -> None:
    fake, _ = _patch_responses(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "searching",
                },
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "weather"},
                },
            },
        ]
    )

    stream = ai.Stream(
        protocol.OpenAIResponsesProtocol().stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            provider="openai",
        )
    )
    seen: list[type[events.Event]] = []
    async for event in stream:
        seen.append(type(event))

    assert events.BuiltinToolStart in seen
    assert events.BuiltinToolEnd in seen
    assert len(stream.message.builtin_tool_calls) == 1
    assert stream.message.builtin_tool_calls[0].tool_name == "web_search"
    assert len(stream.message.builtin_tool_returns) == 1
    assert stream.message.builtin_tool_returns[0].result == {
        "action": {"type": "search", "query": "weather"}
    }


async def test_system_messages_use_openai_system_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captured = _patch(monkeypatch)

    await _drain(
        protocol.stream(
            fake,
            _MODEL,
            [ai.system_message("rules"), ai.user_message("Hi")],
            provider="openai",
        )
    )

    assert captured["messages"][0] == {"role": "system", "content": "rules"}


async def test_params_translate_to_sdk_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captured = _patch(monkeypatch)

    await _drain(
        protocol.stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            params=ai.InferenceRequestParams(
                sampling={
                    ai.TemperatureSamplerParams: ai.TemperatureSamplerParams(
                        temperature=0.2
                    ),
                    ai.TopPSamplerParams: ai.TopPSamplerParams(top_p=0.9),
                    ai.SeedSamplerParams: ai.SeedSamplerParams(seed=123),
                },
                output=ai.OutputParams(max_tokens=128),
                provider_service=ai.ProviderServiceParams(service_tier="auto"),
                extra_body={"future_option": True, "verbosity": "low"},
                extra_headers={"x-openai-feature": "enabled"},
            ),
            provider="openai",
        )
    )

    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.9
    assert captured["seed"] == 123
    assert captured["max_completion_tokens"] == 128
    assert captured["service_tier"] == "auto"
    assert captured["extra_body"] == {"future_option": True, "verbosity": "low"}
    assert captured["extra_headers"] == {"x-openai-feature": "enabled"}


async def test_chat_omits_explicit_random_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captured = _patch(monkeypatch)

    await _drain(
        protocol.stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            params=ai.InferenceRequestParams(
                sampling={
                    ai.SeedSamplerParams: ai.SeedSamplerParams(seed=ai.RANDOM)
                }
            ),
            provider="openai",
        )
    )

    assert "seed" not in captured


async def test_responses_rejects_seed() -> None:
    fake, _ = _patch_responses()

    with pytest.raises(ValueError, match="seed"):
        await _drain(
            protocol.OpenAIResponsesProtocol().stream(
                fake,
                _MODEL,
                [ai.user_message("Hi")],
                params=ai.InferenceRequestParams(
                    sampling={
                        ai.SeedSamplerParams: ai.SeedSamplerParams(seed=123)
                    }
                ),
                provider="openai",
            )
        )


async def test_strict_json_schema_flows_into_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captured = _patch(monkeypatch)

    await _drain(
        protocol.stream(
            fake,
            _MODEL,
            [ai.user_message("Hi")],
            output_type=_Answer,
            provider="openai",
        )
    )

    assert captured["response_format"]["json_schema"]["strict"] is True


async def test_non_inference_params_rejected_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = _patch(monkeypatch)

    stream = protocol.stream(
        fake,
        _MODEL,
        [ai.user_message("Hi")],
        params=cast(Any, [{"reasoning_effort": "high"}]),
        provider="openai",
    )

    with pytest.raises(TypeError, match="InferenceRequestParams"):
        await _drain(stream)


async def test_builtin_tool_in_request_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chat-completions rejects OpenAI built-in tools at the boundary."""
    fake, _ = _patch(monkeypatch)

    stream = protocol.stream(
        fake,
        _MODEL,
        [ai.user_message("Hi")],
        tools=[openai_tools.web_search()],
        provider="openai",
    )

    with pytest.raises(NotImplementedError, match="Responses API"):
        await _drain(stream)


async def test_builtin_part_in_messages_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BuiltinToolCallPart`` cannot round-trip through chat-completions."""
    fake, _ = _patch(monkeypatch)

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
        await _drain(protocol.stream(fake, _MODEL, convo, provider="openai"))


async def test_sdk_errors_are_mapped_to_provider_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = monkeypatch
    response = httpx.Response(
        429,
        request=httpx.Request(
            "POST", "https://openai.test/v1/chat/completions"
        ),
        headers={"x-request-id": "req-openai"},
    )
    sdk_error = openai.RateLimitError(
        "slow down",
        response=response,
        body={"type": "rate_limit_error", "code": "rate_limit"},
    )
    fake = _RaisingOpenAIClient(sdk_error)

    with pytest.raises(ai.ProviderRateLimitError) as exc_info:
        await _drain(
            protocol.stream(
                cast("openai.AsyncOpenAI", fake),
                _MODEL,
                [ai.user_message("Hi")],
                provider="openai",
            )
        )

    exc = exc_info.value
    assert exc.provider == "openai"
    assert exc.http_context is not None
    assert exc.http_context.status_code == 429
    assert exc.http_context.request is response.request
    assert exc.http_context.response is response
    assert exc.request_id == "req-openai"
    assert exc.__cause__ is sdk_error


async def test_model_404_is_mapped_to_model_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = monkeypatch
    response = httpx.Response(
        404,
        request=httpx.Request(
            "POST", "https://openai.test/v1/chat/completions"
        ),
    )
    sdk_error = openai.NotFoundError(
        "model not found",
        response=response,
        body={"code": "model_not_found", "param": "model"},
    )
    fake = _RaisingOpenAIClient(sdk_error)

    with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
        await _drain(
            protocol.stream(
                cast("openai.AsyncOpenAI", fake),
                _MODEL,
                [ai.user_message("Hi")],
                provider="openai",
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
