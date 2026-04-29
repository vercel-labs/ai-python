"""Integration tests for the AI Gateway v3 streaming adapter.

Every test exercises the real ``stream()`` function with a ``Client``
wired to an ``httpx.MockTransport``, so the full production code path
is covered:

    stream(client, model, messages)
      -> _build_request_body()
      -> httpx POST (mock)
      -> SSE line parsing
      -> _parse_stream_part()
      -> StreamHandler
      -> yield Message
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import httpx
import pytest

import ai
from ai import models
from ai.models.ai_gateway import ai_gateway, errors
from ai.models.core import model as model_
from ai.types import events, messages

from .conftest import mock_client, sse, user_msg

# The ai_gateway __init__.py re-exports `stream` as a function, which
# shadows the module.  Use importlib to get the actual module.
stream_mod = importlib.import_module("ai.models.ai_gateway.stream")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_MODEL = ai_gateway("test-provider/test-model")


async def _collect(
    client: Any,
    msgs: list[messages.Message],
    model: model_.Model = _TEST_MODEL,
    **kwargs: Any,
) -> list[events.Event]:
    """Drain ``stream()`` and return all yielded events."""
    result: list[events.Event] = []
    async for event in stream_mod.stream(client, model, msgs, **kwargs):
        result.append(event)
    return result


async def _final(
    client: Any,
    msgs: list[messages.Message],
    model: model_.Model = _TEST_MODEL,
    **kwargs: Any,
) -> messages.Message:
    """Drain the adapter's event stream and return the aggregated message."""
    s = models.Stream(stream_mod.stream(client, model, msgs, **kwargs))
    async for _ in s:
        pass
    return s.message


# ---------------------------------------------------------------------------
# Streaming: text, reasoning, tool calls
# ---------------------------------------------------------------------------


class TestStreaming:
    async def test_text_stream(self) -> None:
        body = sse(
            {"type": "text-start", "id": "t1"},
            {"type": "text-delta", "id": "t1", "textDelta": "Hello"},
            {"type": "text-delta", "id": "t1", "textDelta": " World"},
            {"type": "text-end", "id": "t1"},
            {
                "type": "finish",
                "finishReason": "stop",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                },
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        client = mock_client(httpx.MockTransport(handler))
        final = await _final(client, [user_msg("Hi")])
        assert final.text == "Hello World"
        assert final.usage is not None
        assert final.usage.input_tokens == 5
        assert final.usage.output_tokens == 2

    async def test_reasoning_then_text(self) -> None:
        body = sse(
            {"type": "reasoning-start", "id": "r1"},
            {"type": "reasoning-delta", "id": "r1", "delta": "think"},
            {"type": "reasoning-end", "id": "r1"},
            {"type": "text-start", "id": "t1"},
            {"type": "text-delta", "id": "t1", "textDelta": "42"},
            {"type": "text-end", "id": "t1"},
            {"type": "finish", "finishReason": "stop", "usage": {}},
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = await _final(mock_client(httpx.MockTransport(handler)), [user_msg("?")])
        assert final.reasoning == "think"
        assert final.text == "42"

    async def test_streaming_tool_call(self) -> None:
        body = sse(
            {
                "type": "tool-input-start",
                "id": "tc-1",
                "toolName": "search",
            },
            {"type": "tool-input-delta", "id": "tc-1", "delta": '{"q":'},
            {"type": "tool-input-delta", "id": "tc-1", "delta": '"hi"}'},
            {"type": "tool-input-end", "id": "tc-1"},
            {
                "type": "finish",
                "finishReason": "tool-calls",
                "usage": {},
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = await _final(
            mock_client(httpx.MockTransport(handler)), [user_msg("search")]
        )
        tc = final.tool_calls
        assert len(tc) == 1
        assert tc[0].tool_name == "search"
        assert tc[0].tool_args == '{"q":"hi"}'

    async def test_inline_file_stream(self) -> None:
        """Models like Gemini-3-pro-image return inline file parts
        alongside text in the language model stream."""
        body = sse(
            {"type": "text-start", "id": "t1"},
            {"type": "text-delta", "id": "t1", "textDelta": "Here is an image:"},
            {"type": "text-end", "id": "t1"},
            {
                "type": "file",
                "id": "f1",
                "mediaType": "image/png",
                "data": "iVBORw0KGgo=",
            },
            {
                "type": "finish",
                "finishReason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = await _final(
            mock_client(httpx.MockTransport(handler)), [user_msg("draw me")]
        )
        assert final.text == "Here is an image:"
        assert len(final.images) == 1
        assert final.images[0].media_type == "image/png"
        assert final.images[0].data == "iVBORw0KGgo="

    async def test_complete_tool_call_part(self) -> None:
        """Non-streaming ``tool-call`` part (one shot) must also work."""
        body = sse(
            {
                "type": "tool-call",
                "toolCallId": "tc-1",
                "toolName": "get_weather",
                "input": {"city": "SF"},
            },
            {
                "type": "finish",
                "finishReason": "tool-calls",
                "usage": {},
            },
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = await _final(
            mock_client(httpx.MockTransport(handler)), [user_msg("weather")]
        )
        assert len(final.tool_calls) == 1
        assert json.loads(final.tool_calls[0].tool_args) == {"city": "SF"}


# ---------------------------------------------------------------------------
# Request: headers, body, tools
# ---------------------------------------------------------------------------


class TestRequest:
    async def test_protocol_headers(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured.update(dict(req.headers))
            return httpx.Response(
                200,
                text=sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        model = ai_gateway("anthropic/claude-sonnet-4")
        client = mock_client(httpx.MockTransport(handler), api_key="sk-test")
        await _collect(client, [user_msg("Hi")], model=model)

        assert captured["authorization"] == "Bearer sk-test"
        assert captured["ai-gateway-protocol-version"] == "0.0.1"
        assert captured["ai-language-model-specification-version"] == "3"
        assert captured["ai-language-model-id"] == "anthropic/claude-sonnet-4"
        assert captured["ai-language-model-streaming"] == "true"
        assert captured["ai-gateway-auth-method"] == "api-key"

    async def test_body_prompt_format(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        await _collect(mock_client(httpx.MockTransport(handler)), [user_msg("Hello")])

        assert captured_body["prompt"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]

    async def test_provider_options_in_body(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        opts = {"gateway": {"order": ["bedrock", "openai"]}}
        await _collect(
            mock_client(httpx.MockTransport(handler)),
            [user_msg("Hi")],
            provider_options=opts,
        )

        assert captured_body["providerOptions"] == opts

    async def test_real_tool_in_request_body(self) -> None:
        """A real ``@tool``-decorated function must appear correctly
        in the request body sent to the gateway."""

        @ai.tool
        async def lookup(query: str) -> str:
            """Search the database."""
            return "result"

        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        await _collect(
            mock_client(httpx.MockTransport(handler)),
            [user_msg("find something")],
            tools=[lookup],
        )

        assert "tools" in captured_body
        td = captured_body["tools"][0]
        assert td["name"] == "lookup"
        assert td["type"] == "function"
        assert "query" in td["inputSchema"]["properties"]

    async def test_multi_turn_request_body(self) -> None:
        """A multi-turn conversation including a tool result must
        serialize correctly into the v3 prompt format."""
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse({"type": "finish", "finishReason": "stop", "usage": {}}),
            )

        tool_call = messages.ToolCallPart(
            tool_call_id="tc-1",
            tool_name="search",
            tool_args='{"q": "weather"}',
        )
        tool_result = messages.ToolResultPart(
            tool_call_id="tc-1",
            tool_name="search",
            result={"temp": 72},
        )
        conversation = [
            user_msg("What's the weather?"),
            messages.Message(role="assistant", parts=[tool_call]),
            messages.Message(role="tool", parts=[tool_result]),
            user_msg("Thanks, and tomorrow?"),
        ]

        await _collect(mock_client(httpx.MockTransport(handler)), conversation)

        prompt = captured_body["prompt"]
        # user -> assistant (tool-call) -> tool (tool-result) -> user
        assert len(prompt) == 4
        assert prompt[0]["role"] == "user"
        assert prompt[1]["role"] == "assistant"
        assert prompt[1]["content"][0]["type"] == "tool-call"
        assert prompt[2]["role"] == "tool"
        assert prompt[2]["content"][0]["type"] == "tool-result"
        assert prompt[3]["role"] == "user"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_401_authentication_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                    }
                },
            )

        with pytest.raises(errors.GatewayAuthenticationError):
            await _collect(mock_client(httpx.MockTransport(handler)), [user_msg("Hi")])

    async def test_429_rate_limit_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_exceeded",
                    }
                },
            )

        with pytest.raises(errors.GatewayRateLimitError):
            await _collect(mock_client(httpx.MockTransport(handler)), [user_msg("Hi")])

    async def test_404_model_not_found(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": "Model xyz not found",
                        "type": "model_not_found",
                        "param": {"modelId": "xyz"},
                    }
                },
            )

        with pytest.raises(errors.GatewayModelNotFoundError) as exc_info:
            await _collect(mock_client(httpx.MockTransport(handler)), [user_msg("Hi")])
        assert exc_info.value.model_id == "xyz"

    async def test_500_malformed_response(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Not JSON")

        with pytest.raises(errors.GatewayResponseError):
            await _collect(mock_client(httpx.MockTransport(handler)), [user_msg("Hi")])
