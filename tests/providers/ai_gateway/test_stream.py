"""Integration tests for the AI Gateway v3 streaming adapter.

Every test exercises the real ``stream()`` function with a provider
wired to an ``httpx.MockTransport``, so the full production code path
is covered:

    stream(model, messages)
      -> _build_request_body()
      -> httpx POST (mock)
      -> SSE line parsing
      -> _parse_stream_part()
      -> StreamHandler
      -> yield Message
"""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest

import ai
from ai import models
from ai.models.core import model as model_
from ai.providers.ai_gateway import GatewayParams, ProviderTimeoutsParams
from ai.types import events, messages

from .conftest import mock_client, mock_model, sse, user_msg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect(
    model: model_.Model,
    msgs: list[messages.Message],
    **kwargs: Any,
) -> list[events.Event]:
    """Drain ``stream()`` and return all yielded events."""
    result: list[events.Event] = []
    async for event in model.provider.stream(model, msgs, **kwargs):
        result.append(event)
    return result


async def _final(
    model: model_.Model,
    msgs: list[messages.Message],
    **kwargs: Any,
) -> messages.Message:
    """Drain the adapter's event stream and return the aggregated message."""
    s = models.Stream(model.provider.stream(model, msgs, **kwargs))
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

        final = await _final(
            mock_client(httpx.MockTransport(handler)), [user_msg("?")]
        )
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
            {
                "type": "text-delta",
                "id": "t1",
                "textDelta": "Here is an image:",
            },
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

    async def test_provider_executed_tool_call_streaming(self) -> None:
        """Route ``tool-input-*`` to BuiltinTool* events.

        ``tool-result`` with ``providerExecuted: true`` aggregates into
        ``Message.builtin_tool_returns``.
        """
        result_payload = [{"title": "Forecast", "url": "https://example.com"}]
        body = sse(
            {
                "type": "tool-input-start",
                "id": "tc-1",
                "toolName": "web_search",
                "providerExecuted": True,
            },
            {
                "type": "tool-input-delta",
                "id": "tc-1",
                "delta": '{"q":"weather"}',
            },
            {"type": "tool-input-end", "id": "tc-1"},
            {
                "type": "tool-result",
                "toolCallId": "tc-1",
                "toolName": "web_search",
                "output": result_payload,
                "providerExecuted": True,
            },
            {"type": "finish", "finishReason": "stop", "usage": {}},
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        final = await _final(
            mock_client(httpx.MockTransport(handler)), [user_msg("weather")]
        )

        assert len(final.builtin_tool_calls) == 1
        bt = final.builtin_tool_calls[0]
        assert bt.tool_call_id == "tc-1"
        assert bt.tool_args == '{"q":"weather"}'

        assert len(final.builtin_tool_returns) == 1
        ret = final.builtin_tool_returns[0]
        assert ret.tool_call_id == "tc-1"
        assert ret.tool_name == "web_search"
        assert ret.result == result_payload

    async def test_provider_executed_one_shot_tool_call(self) -> None:
        """Expand provider-executed one-shot calls to BuiltinTool*."""
        body = sse(
            {
                "type": "tool-call",
                "toolCallId": "tc-1",
                "toolName": "web_search",
                "input": {"q": "x"},
                "providerExecuted": True,
            },
            {"type": "finish", "finishReason": "stop", "usage": {}},
        )

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body)

        events_seen: list[type] = []
        model = mock_client(httpx.MockTransport(handler))
        async for event in model.provider.stream(model, [user_msg("hi")]):
            events_seen.append(type(event))

        assert events.BuiltinToolStart in events_seen
        assert events.BuiltinToolDelta in events_seen
        assert events.BuiltinToolEnd in events_seen
        # Crucially, no host-tool events for a provider-executed call.
        assert events.ToolStart not in events_seen
        assert events.ToolEnd not in events_seen


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
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        model = mock_model(
            httpx.MockTransport(handler),
            api_key="sk-test",
            model_id="anthropic/claude-sonnet-4",
        )
        await _collect(model, [user_msg("Hi")])

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
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        await _collect(
            mock_client(httpx.MockTransport(handler)), [user_msg("Hello")]
        )

        assert captured_body["prompt"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]

    async def test_gateway_params_pass_through_as_raw_body(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        model = mock_model(
            httpx.MockTransport(handler),
            model_id="anthropic/claude-sonnet-4",
        )
        request_params = ai.InferenceRequestParams(
            sampling={ai.SeedSamplerParams: ai.SeedSamplerParams(seed=123)},
            output=ai.OutputParams(reasoning_summary="detailed"),
            reasoning=ai.ReasoningParams(effort="high"),
            context_management=ai.ContextManagementParams(
                compaction=ai.TokenThreshold(120_000)
            ),
            routing=ai.RoutingParams(provider_order=("bedrock", "anthropic")),
            provider_params={
                GatewayParams: GatewayParams(zero_data_retention=True)
            },
            extra_body={
                "providerOptions": {
                    "anthropic": {
                        "speed": "fast",
                        "futureAnthropicField": True,
                    },
                    "google": {
                        "thinkingConfig": {"budgetTokens": 1024},
                    },
                },
                "futureGatewayField": True,
            },
        )
        async with models.stream(
            model,
            [user_msg("Hi")],
            params=request_params,
        ) as stream:
            async for _ in stream:
                pass

        assert captured_body["providerOptions"] == {
            "gateway": {
                "order": ["bedrock", "anthropic"],
                "zeroDataRetention": True,
            },
            "anthropic": {
                "effort": "high",
                "speed": "fast",
                "futureAnthropicField": True,
                "contextManagement": {
                    "edits": [
                        {
                            "type": "compact_20260112",
                            "trigger": {
                                "type": "input_tokens",
                                "value": 120_000,
                            },
                        }
                    ]
                },
                "thinking": {"type": "adaptive", "display": "detailed"},
            },
            "google": {
                "thinkingConfig": {"budgetTokens": 1024},
            },
        }
        assert captured_body["seed"] == 123
        assert captured_body["futureGatewayField"] is True

    async def test_gateway_anthropic_effort_enables_thinking(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        model = mock_model(
            httpx.MockTransport(handler),
            model_id="anthropic/claude-opus-4-8",
        )
        request_params = ai.InferenceRequestParams(
            reasoning=ai.ReasoningParams(effort="high"),
        )
        async with models.stream(
            model,
            [user_msg("Hi")],
            params=request_params,
        ) as stream:
            async for _ in stream:
                pass

        # `effort` alone is a no-op upstream; the gateway only turns
        # thinking on when a `thinking` block is also present.
        assert captured_body["providerOptions"]["anthropic"] == {
            "effort": "high",
            "thinking": {"type": "adaptive"},
        }

    async def test_gateway_anthropic_summary_none_keeps_thinking(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        model = mock_model(
            httpx.MockTransport(handler),
            model_id="anthropic/claude-opus-4-8",
        )
        request_params = ai.InferenceRequestParams(
            reasoning=ai.ReasoningParams(effort="high"),
            output=ai.OutputParams(reasoning_summary=None),
        )
        async with models.stream(
            model,
            [user_msg("Hi")],
            params=request_params,
        ) as stream:
            async for _ in stream:
                pass

        # reasoning_summary=None omits the summary but keeps thinking on.
        assert captured_body["providerOptions"]["anthropic"] == {
            "effort": "high",
            "thinking": {"type": "adaptive", "display": "omitted"},
        }

    async def test_gateway_omits_random_seed(self) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        model = mock_model(
            httpx.MockTransport(handler),
            model_id="openai/gpt-5.4",
        )
        request_params = ai.InferenceRequestParams(
            sampling={ai.SeedSamplerParams: ai.SeedSamplerParams(seed=-1)}
        )
        async with models.stream(
            model,
            [user_msg("Hi")],
            params=request_params,
        ) as stream:
            async for _ in stream:
                pass

        assert "seed" not in captured_body

    async def test_gateway_routing_params_map_to_provider_options(
        self,
    ) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        model = mock_model(
            httpx.MockTransport(handler),
            model_id="anthropic/claude-sonnet-4",
        )
        byok: dict[str, list[dict[str, Any]]] = {
            "anthropic": [{"apiKey": "sk-test"}]
        }
        request_params = ai.InferenceRequestParams(
            cache=ai.CacheParams(mode="auto"),
            safety_identifier="user_123",
            tags=frozenset({"team:search", "env:prod"}),
            routing=ai.RoutingParams(
                provider_allowlist=frozenset({"anthropic", "bedrock"}),
                provider_order=("bedrock", "anthropic"),
                provider_ranking=ai.ProviderRankingStrategy.LATENCY,
                fallback_models=("openai/gpt-5-mini",),
                routing_target=ai.CloudRegion("us-east-1"),
            ),
            provider_params={
                GatewayParams: GatewayParams(
                    quota_entity_id="quota_123",
                    zero_data_retention=True,
                    hipaa_compliant=True,
                    disallow_prompt_training=True,
                    byok=byok,
                    provider_timeouts=ProviderTimeoutsParams(
                        byok={"anthropic": 5000}
                    ),
                ),
            },
        )
        async with models.stream(
            model,
            [user_msg("Hi")],
            params=request_params,
        ) as stream:
            async for _ in stream:
                pass

        assert captured_body["providerOptions"]["gateway"] == {
            "caching": "auto",
            "only": ["anthropic", "bedrock"],
            "order": ["bedrock", "anthropic"],
            "sort": "latency",
            "models": ["openai/gpt-5-mini"],
            "user": "user_123",
            "tags": ["env:prod", "team:search"],
            "quotaEntityId": "quota_123",
            "zeroDataRetention": True,
            "hipaaCompliant": True,
            "disallowPromptTraining": True,
            "inferenceRegion": {"providerRegion": "us-east-1"},
            "byok": {"anthropic": [{"apiKey": "sk-test"}]},
            "providerTimeouts": {"byok": {"anthropic": 5000}},
        }

    async def test_gateway_openai_context_management_maps_to_provider_options(
        self,
    ) -> None:
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        model = mock_model(
            httpx.MockTransport(handler),
            model_id="openai/gpt-5.4",
        )
        request_params = ai.InferenceRequestParams(
            context_management=ai.ContextManagementParams(
                compaction=ai.TokenThreshold(120_000)
            )
        )
        async with models.stream(
            model,
            [user_msg("Hi")],
            params=request_params,
        ) as stream:
            async for _ in stream:
                pass

        assert captured_body["providerOptions"]["openai"] == {
            "contextManagement": [
                {"type": "compaction", "compactThreshold": 120_000}
            ]
        }

    async def test_gateway_rejects_non_dict_params(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            raise AssertionError("request should not be sent")

        model = mock_model(
            httpx.MockTransport(handler),
            model_id="openai/gpt-5.4",
        )
        with pytest.raises(TypeError, match="InferenceRequestParams"):
            async with models.stream(
                model,
                [user_msg("Hi")],
                params=cast(
                    Any,
                    [{"providerOptions": {"openai": {"serviceTier": "auto"}}}],
                ),
            ) as stream:
                async for _ in stream:
                    pass

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
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        await _collect(
            mock_client(httpx.MockTransport(handler)),
            [user_msg("find something")],
            tools=[lookup.tool],
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
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
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

    async def test_multi_turn_round_trip_builtin_parts(self) -> None:
        """``BuiltinToolCallPart``/``BuiltinToolReturnPart`` serialize as v3
        ``tool-call``/``tool-result`` blocks.
        """
        captured_body: dict[str, Any] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(req.content))
            return httpx.Response(
                200,
                text=sse(
                    {"type": "finish", "finishReason": "stop", "usage": {}}
                ),
            )

        call = messages.BuiltinToolCallPart(
            tool_call_id="srvtoolu_1",
            tool_name="web_search",
            tool_args='{"q":"weather"}',
        )
        ret = messages.BuiltinToolReturnPart(
            tool_call_id="srvtoolu_1",
            tool_name="web_search",
            result=[{"title": "Forecast"}],
        )
        convo = [
            user_msg("weather?"),
            messages.Message(role="assistant", parts=[call, ret]),
            user_msg("thanks"),
        ]

        await _collect(mock_client(httpx.MockTransport(handler)), convo)

        assistant = next(
            m for m in captured_body["prompt"] if m["role"] == "assistant"
        )
        assert assistant["content"] == [
            {
                "type": "tool-call",
                "toolCallId": "srvtoolu_1",
                "toolName": "web_search",
                "input": {"q": "weather"},
                "providerExecuted": True,
            },
            {
                "type": "tool-result",
                "toolCallId": "srvtoolu_1",
                "toolName": "web_search",
                "output": {
                    "type": "json",
                    "value": [{"title": "Forecast"}],
                },
                "providerExecuted": True,
            },
        ]


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

        with pytest.raises(ai.ProviderAuthenticationError):
            await _collect(
                mock_client(httpx.MockTransport(handler)), [user_msg("Hi")]
            )

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

        with pytest.raises(ai.ProviderRateLimitError):
            await _collect(
                mock_client(httpx.MockTransport(handler)), [user_msg("Hi")]
            )

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

        with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
            await _collect(
                mock_client(httpx.MockTransport(handler)), [user_msg("Hi")]
            )
        assert exc_info.value.model_id == "xyz"

    async def test_500_malformed_response(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Not JSON")

        with pytest.raises(ai.ProviderResponseError):
            await _collect(
                mock_client(httpx.MockTransport(handler)), [user_msg("Hi")]
            )
