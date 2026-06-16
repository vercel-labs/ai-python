"""Anthropic ``probe`` tests.

The status-code handling is shared with OpenAI and exhaustively tested
in ``tests/providers/openai/test_probe.py``. This file only confirms the
provider-specific 200 path so we know URL routing is wired up.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx
import pytest

import ai
from ai.providers.anthropic import AnthropicCompatibleProvider


def probe_provider(
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
    base_url: str = "https://anthropic.test",
) -> ai.Provider[Any]:
    """Anthropic provider whose mock response is built from JSON args."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(json_body or {}).encode()
        return httpx.Response(status_code, content=body)

    return ai.get_provider(
        "anthropic",
        base_url=base_url,
        api_key="sk-test-key",
        client=httpx.AsyncClient(
            base_url=base_url,
            transport=httpx.MockTransport(_handler),
        ),
    )


def _client_with_mock(
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
    base_url: str = "https://anthropic.test",
) -> ai.Model:
    return ai.Model(
        "claude-opus-4-6",
        provider_factory=probe_provider,
        provider_args={
            "status_code": status_code,
            "json_body": json_body,
            "base_url": base_url,
        },
    )


async def test_200_succeeds() -> None:
    model = _client_with_mock(200, {"id": "claude-opus-4-6", "type": "model"})
    await model.provider.probe(model)


async def test_model_not_found_raises_model_not_found() -> None:
    model = _client_with_mock(404)
    with pytest.raises(ai.ProviderModelNotFoundError) as exc_info:
        await model.provider.probe(model)

    assert exc_info.value.model_id == model.id


class _HeaderCaptureProvider(AnthropicCompatibleProvider):
    """Custom provider that records request headers.

    A provider subclass is itself a valid model factory: the class is
    module-level, so ``ai.Model`` can serialize a reference to it, and
    per-instance state (the captured headers) stays on the provider,
    reachable through ``model.provider``.
    """

    handles: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        self.captured_headers: dict[str, str] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            self.captured_headers.update(dict(request.headers))
            body = json.dumps({"id": "custom-model", "type": "model"}).encode()
            return httpx.Response(200, content=body)

        super().__init__(
            name="custom-anthropic",
            default_base_url="https://anthropic.test",
            api_key="sk-test-key",
            anthropic_version="2024-01-01",
            headers={"X-Custom-Header": "example"},
            client=httpx.AsyncClient(
                base_url="https://anthropic.test",
                transport=httpx.MockTransport(_handler),
            ),
        )


async def test_custom_anthropic_version_header() -> None:
    model = ai.Model("custom-model", provider_factory=_HeaderCaptureProvider)

    provider = model.provider
    assert isinstance(provider, _HeaderCaptureProvider)
    await provider.probe(model)
    assert provider.captured_headers["anthropic-version"] == "2024-01-01"
    assert provider.captured_headers["x-custom-header"] == "example"
