"""Anthropic ``probe`` tests.

The status-code handling is shared with OpenAI and exhaustively tested
in ``tests/providers/openai/test_probe.py``. This file only confirms the
provider-specific 200 path so we know URL routing is wired up.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import httpx
import pydantic
import pytest

import ai
from ai.providers.anthropic import AnthropicCompatibleProvider


class _ProbeProviderRef(ai.ProviderRef):
    status_code: int = 200
    json_body: dict[str, Any] | None = None

    def __init__(
        self,
        status_code: int = 200,
        json_body: dict[str, Any] | None = None,
        base_url: str = "https://anthropic.test",
    ) -> None:
        super().__init__(
            "anthropic",
            status_code=status_code,
            json_body=json_body,
            base_url=base_url,
        )

    def build(self) -> ai.Provider[Any]:
        def _handler(request: httpx.Request) -> httpx.Response:
            _ = request
            body = json.dumps(self.json_body or {}).encode()
            return httpx.Response(self.status_code, content=body)

        assert self.base_url is not None
        return ai.get_provider(
            "anthropic",
            base_url=self.base_url,
            api_key="sk-test-key",
            client=httpx.AsyncClient(
                base_url=self.base_url,
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
        provider_ref=_ProbeProviderRef(status_code, json_body, base_url),
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
    """Custom provider that records request headers."""

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


class _HeaderCaptureProviderRef(ai.ProviderRef):
    _provider: _HeaderCaptureProvider = pydantic.PrivateAttr()

    def __init__(self) -> None:
        super().__init__("anthropic")
        self._provider = _HeaderCaptureProvider()

    def build(self) -> _HeaderCaptureProvider:
        return self._provider


async def test_custom_anthropic_version_header() -> None:
    model = ai.Model("custom-model", provider_ref=_HeaderCaptureProviderRef())

    provider = model.provider
    assert isinstance(provider, _HeaderCaptureProvider)
    await provider.probe(model)
    assert provider.captured_headers["anthropic-version"] == "2024-01-01"
    assert provider.captured_headers["x-custom-header"] == "example"
