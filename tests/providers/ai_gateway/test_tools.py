"""Tests for the AI Gateway built-in tool surface.

The gateway adapter accepts native ``anthropic.tools.*``, ``openai.tools.*``,
and ``ai_gateway.tools.*`` instances and serializes them as v3 ``provider``
blocks of the shape ``{type, id, name, args}`` with camelCase ``args`` keys.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pydantic
import pytest

from ai import types
from ai.providers.ai_gateway import adapter
from ai.providers.ai_gateway import tools as gateway_tools
from ai.providers.anthropic import tools as anthropic_tools
from ai.providers.openai import tools as openai_tools

from .conftest import mock_model, sse, user_msg


def _capture_body_handler(captured: dict[str, Any]) -> Any:
    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            text=sse({"type": "finish", "finishReason": "stop", "usage": {}}),
        )

    return handler


class TestGatewayBuiltins:
    async def test_anthropic_web_search_serializes(self) -> None:
        captured: dict[str, Any] = {}
        model = mock_model(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            model,
            [user_msg("hi")],
            tools=[anthropic_tools.web_search(max_uses=3)],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "anthropic.web_search_20260209",
                "name": "web_search",
                "args": {"maxUses": 3},
            }
        ]

    async def test_openai_mcp_serializes(self) -> None:
        captured: dict[str, Any] = {}
        model = mock_model(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            model,
            [user_msg("hi")],
            tools=[
                openai_tools.mcp(
                    server_label="my-server",
                    server_url="https://mcp.example.com",
                ),
            ],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "openai.mcp",
                "name": "mcp",
                "args": {
                    "serverLabel": "my-server",
                    "serverUrl": "https://mcp.example.com",
                },
            }
        ]

    async def test_gateway_perplexity_search_serializes(self) -> None:
        captured: dict[str, Any] = {}
        model = mock_model(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            model,
            [user_msg("hi")],
            tools=[
                gateway_tools.perplexity_search(
                    max_results=5,
                    search_domain_filter=["nature.com"],
                ),
            ],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "gateway.perplexity_search",
                "name": "perplexity_search",
                "args": {
                    "maxResults": 5,
                    "searchDomainFilter": ["nature.com"],
                },
            }
        ]

    async def test_gateway_parallel_search_serializes(self) -> None:
        captured: dict[str, Any] = {}
        model = mock_model(httpx.MockTransport(_capture_body_handler(captured)))

        async for _ in adapter.stream(
            model,
            [user_msg("hi")],
            tools=[
                gateway_tools.parallel_search(
                    mode="agentic",
                    source_policy=gateway_tools.SourcePolicy(
                        include_domains=["wikipedia.org"],
                    ),
                ),
            ],
        ):
            pass

        assert captured["tools"] == [
            {
                "type": "provider",
                "id": "gateway.parallel_search",
                "name": "parallel_search",
                "args": {
                    "mode": "agentic",
                    "sourcePolicy": {"includeDomains": ["wikipedia.org"]},
                },
            }
        ]

    async def test_unknown_provider_args_rejected(self) -> None:
        """Provider-executed tools need a registered args type."""

        class UnknownArgs(pydantic.BaseModel):
            value: str

        def handler(req: httpx.Request) -> httpx.Response:
            raise AssertionError("request should not be sent")

        stream = adapter.stream(
            mock_model(httpx.MockTransport(handler)),
            [user_msg("hi")],
            tools=[
                types.tools.Tool(
                    kind="provider",
                    name="bad",
                    args=UnknownArgs(value="x"),
                )
            ],
        )

        with pytest.raises(TypeError, match="unsupported args"):
            async for _ in stream:
                pass
