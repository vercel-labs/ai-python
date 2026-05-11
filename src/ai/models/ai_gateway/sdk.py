"""AI Gateway v4 HTTP API"""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .. import core
from . import errors

_PROTOCOL_VERSION = "0.0.1"
_MODEL_SPECIFICATION_VERSION = "4"

ModelType = Literal["language", "image", "video", "embedding", "reranking"]


class GatewayClient:
    def __init__(
        self,
        client: core.client.Client,
        model: core.model.Model | None = None,
    ) -> None:
        self._client = client
        self._model = model

    @property
    def base_url(self) -> str:
        return self._client.base_url.rstrip("/")

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def origin_url(self, path: str) -> str:
        parsed = urlparse(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}/{path.lstrip('/')}"

    def protocol_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "ai-gateway-protocol-version": _PROTOCOL_VERSION,
        }
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"
            headers["ai-gateway-auth-method"] = "api-key"
        return headers

    def model_headers(
        self,
        model_type: ModelType,
        *,
        streaming: bool = False,
        accept: str | None = None,
    ) -> dict[str, str]:
        if self._model is None:
            raise ValueError("Gateway model headers require a model.")

        headers = {
            "Content-Type": "application/json",
            **self.protocol_headers(),
        }

        if model_type == "language":
            headers["ai-language-model-specification-version"] = (
                _MODEL_SPECIFICATION_VERSION
            )
            headers["ai-language-model-id"] = self._model.id
            headers["ai-language-model-streaming"] = str(streaming).lower()
        elif model_type == "image":
            headers["ai-image-model-specification-version"] = (
                _MODEL_SPECIFICATION_VERSION
            )
            headers["ai-model-id"] = self._model.id
        elif model_type == "video":
            headers["ai-video-model-specification-version"] = (
                _MODEL_SPECIFICATION_VERSION
            )
            headers["ai-model-id"] = self._model.id
        elif model_type == "embedding":
            headers["ai-embedding-model-specification-version"] = (
                _MODEL_SPECIFICATION_VERSION
            )
            headers["ai-model-id"] = self._model.id
        elif model_type == "reranking":
            headers["ai-reranking-model-specification-version"] = (
                _MODEL_SPECIFICATION_VERSION
            )
            headers["ai-model-id"] = self._model.id

        if accept is not None:
            headers["accept"] = accept

        return headers

    async def get(
        self,
        path: str,
        *,
        origin: bool = False,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = self.origin_url(path) if origin else self.url(path)
        return await self._client.http.get(
            url,
            headers=headers or self.protocol_headers(),
        )

    async def post_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        model_type: ModelType,
        timeout: httpx.Timeout | float | None = None,
    ) -> httpx.Response:
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout

        response = await self._client.http.post(
            self.url(path),
            json=body,
            headers=self.model_headers(model_type),
            **kwargs,
        )
        await self.raise_for_error(response)
        return response

    @asynccontextmanager
    async def stream(
        self,
        path: str,
        body: dict[str, Any],
        *,
        model_type: ModelType,
        streaming: bool = False,
        accept: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        request_headers = self.model_headers(
            model_type,
            streaming=streaming,
            accept=accept,
        )
        if headers:
            request_headers.update(headers)

        async with self._client.http.stream(
            "POST",
            self.url(path),
            json=body,
            headers=request_headers,
            **kwargs,
        ) as response:
            await self.raise_for_error(response)
            yield response

    async def raise_for_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        await response.aread()
        raise errors.create_gateway_error(
            response_body=response.text,
            status_code=response.status_code,
            api_key_provided=bool(self._client.api_key),
        )

    async def iter_sse(
        self,
        response: httpx.Response,
    ) -> AsyncGenerator[dict[str, Any]]:
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
