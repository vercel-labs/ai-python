"""Minimal async AI Gateway client used by the provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx

from . import errors

if TYPE_CHECKING:
    from ....models.core import model as model_

_PROTOCOL_VERSION = "0.0.1"

ModelType = Literal["language", "image", "video"]


class GatewayClient:
    """Small async HTTP client for Gateway provider endpoints.

    This intentionally implements only the calls used by the current provider:
    config/credits reads, language streaming, image JSON generation, and video
    SSE generation.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.headers = dict(headers or {})
        self._http = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=300.0, connect=10.0),
        )
        self._owns_http = client is None

    async def aclose(self) -> None:
        if self._owns_http and not self._http.is_closed:
            await self._http.aclose()

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def origin_url(self, path: str) -> str:
        parsed = urlparse(self.base_url.rstrip("/"))
        return f"{parsed.scheme}://{parsed.netloc}/{path.lstrip('/')}"

    def protocol_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        headers["ai-gateway-protocol-version"] = _PROTOCOL_VERSION
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["ai-gateway-auth-method"] = "api-key"
        return headers

    def model_headers(
        self,
        model: model_.Model,
        model_type: ModelType,
        *,
        streaming: bool = False,
        accept: str | None = None,
    ) -> dict[str, str]:
        headers = self.protocol_headers()
        headers["Content-Type"] = "application/json"

        if model_type == "language":
            headers["ai-language-model-specification-version"] = "3"
            headers["ai-language-model-id"] = model.id
            headers["ai-language-model-streaming"] = str(streaming).lower()
        elif model_type == "image":
            headers["ai-image-model-specification-version"] = "3"
            headers["ai-model-id"] = model.id
        elif model_type == "video":
            headers["ai-video-model-specification-version"] = "3"
            headers["ai-model-id"] = model.id

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
        request_headers = self.protocol_headers()
        if headers:
            request_headers.update(headers)
        try:
            return await self._http.get(
                url,
                headers=request_headers,
            )
        except httpx.TimeoutException as exc:
            raise errors.GatewayTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise errors.GatewayResponseError(
                message=f"Gateway request failed: {exc}",
            ) from exc

    async def post_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        model: model_.Model,
        model_type: ModelType,
        timeout: httpx.Timeout | float | None = None,
    ) -> httpx.Response:
        try:
            if timeout is None:
                response = await self._http.post(
                    self.url(path),
                    json=body,
                    headers=self.model_headers(model, model_type),
                )
            else:
                response = await self._http.post(
                    self.url(path),
                    json=body,
                    headers=self.model_headers(model, model_type),
                    timeout=timeout,
                )
        except httpx.TimeoutException as exc:
            raise errors.GatewayTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise errors.GatewayResponseError(
                message=f"Gateway request failed: {exc}",
            ) from exc
        await self.raise_for_error(response)
        return response

    async def list_model_ids(self) -> list[str]:
        """List available model IDs from the Gateway config endpoint."""
        response = await self.get("config")
        await self.raise_for_error(response)
        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise errors.GatewayResponseError(
                "Invalid Gateway config response",
                status_code=response.status_code,
                response_body=response.text,
            ) from exc
        return sorted(str(m["id"]) for m in data.get("models", []))

    async def probe_model(self, model_id: str) -> None:
        """Raise unless auth succeeds and ``model_id`` is available."""
        auth_resp = await self.get("v1/credits", origin=True)
        if auth_resp.status_code in {401, 403}:
            raise errors.GatewayAuthenticationError.create_contextual(
                api_key_provided=bool(self.api_key),
                status_code=auth_resp.status_code,
            )
        if auth_resp.status_code != 200:
            await self.raise_for_error(auth_resp)

        remote_ids = set(await self.list_model_ids())
        if model_id not in remote_ids:
            raise errors.GatewayModelNotFoundError(
                f"Model {model_id!r} not found",
                model_id=model_id,
            )

    @asynccontextmanager
    async def stream(
        self,
        path: str,
        body: dict[str, Any],
        *,
        model: model_.Model,
        model_type: ModelType,
        streaming: bool = False,
        accept: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | float | None = None,
    ) -> AsyncIterator[httpx.Response]:
        request_headers = self.model_headers(
            model,
            model_type,
            streaming=streaming,
            accept=accept,
        )
        if headers:
            request_headers.update(headers)

        stream = (
            self._http.stream(
                "POST",
                self.url(path),
                json=body,
                headers=request_headers,
            )
            if timeout is None
            else self._http.stream(
                "POST",
                self.url(path),
                json=body,
                headers=request_headers,
                timeout=timeout,
            )
        )

        try:
            async with stream as response:
                await self.raise_for_error(response)
                yield response
        except httpx.TimeoutException as exc:
            raise errors.GatewayTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise errors.GatewayResponseError(
                message=f"Gateway request failed: {exc}",
            ) from exc

    async def raise_for_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        await response.aread()
        raise errors.create_gateway_error(
            response_body=response.text,
            status_code=response.status_code,
            api_key_provided=bool(self.api_key),
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
                value = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value
