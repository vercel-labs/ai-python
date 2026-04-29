"""AI Gateway v3 generation adapter — image-model and video-model endpoints.

Unified :func:`generate` entry point that dispatches based on param type.
"""

from __future__ import annotations

from typing import Any

import httpx

from ...types import media
from ...types import messages as messages_
from ..core import client as client_
from ..core import model as model_
from ..core.helpers import files
from ..core.params import GenerateParams as GenerateParams
from ..core.params import ImageParams as ImageParams
from ..core.params import VideoParams as VideoParams
from . import _common, errors

# ---------------------------------------------------------------------------
# Image generation — /image-model
# ---------------------------------------------------------------------------


async def _generate_image(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: ImageParams,
) -> messages_.Message:
    """Hit ``/image-model`` and return a Message with FileParts."""
    prompt = _common.extract_prompt(messages)
    input_files = _common.extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["files"] = [_common.file_part_to_wire(f) for f in input_files]

    url = f"{client.base_url.rstrip('/')}/image-model"
    headers = _common.request_headers(client, model, model_type="image")

    response = await client.http.post(url, json=body, headers=headers)
    if response.status_code >= 400:
        raise errors.create_gateway_error(
            response_body=response.text,
            status_code=response.status_code,
            api_key_provided=bool(client.api_key),
        )

    data = response.json()
    raw_images: list[str] = data.get("images", [])
    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = messages_.Usage(
            input_tokens=usage_data.get("inputTokens") or 0,
            output_tokens=usage_data.get("outputTokens") or 0,
        )

    parts: list[messages_.Part] = []
    for img_b64 in raw_images:
        media_type = media.detect_image_media_type(img_b64) or "image/png"
        parts.append(messages_.FilePart(data=img_b64, media_type=media_type))

    return messages_.Message(role="assistant", parts=parts, usage=usage)


# ---------------------------------------------------------------------------
# Video generation — /video-model (SSE response)
# ---------------------------------------------------------------------------


async def _generate_video(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: VideoParams,
) -> messages_.Message:
    """Hit ``/video-model`` (SSE) and return a Message with FileParts."""
    prompt = _common.extract_prompt(messages)
    input_files = _common.extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["image"] = _common.file_part_to_wire(input_files[0])

    url = f"{client.base_url.rstrip('/')}/video-model"
    headers = _common.request_headers(client, model, model_type="video")
    headers["accept"] = "text/event-stream"

    async with client.http.stream(
        "POST",
        url,
        json=body,
        headers=headers,
        timeout=httpx.Timeout(timeout=600.0, connect=10.0),
    ) as response:
        if response.status_code >= 400:
            await response.aread()
            raise errors.create_gateway_error(
                response_body=response.text,
                status_code=response.status_code,
                api_key_provided=bool(client.api_key),
            )

        # Read first SSE data event — the gateway sends a single result event.
        event_data: dict[str, Any] = {}
        async for parsed in _common.parse_sse_lines(response):
            event_data = parsed
            break

    if not event_data:
        raise errors.GatewayResponseError(
            "SSE stream ended without any data events",
        )

    if event_data.get("type") == "error":
        raise errors.GatewayInvalidRequestError(
            message=event_data.get("message", "unknown error"),
            status_code=event_data.get("statusCode", 400),
        )

    raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
    parts: list[messages_.Part] = []
    for video_data in raw_videos:
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            downloaded_bytes, content_type = await files.download(video_data["url"])
            if content_type:
                media_type = content_type
            parts.append(
                messages_.FilePart(data=downloaded_bytes, media_type=media_type)
            )
        else:
            raw_data = video_data.get("data", "")
            parts.append(messages_.FilePart(data=raw_data, media_type=media_type))

    return messages_.Message(role="assistant", parts=parts)


# ---------------------------------------------------------------------------
# Public adapter function
# ---------------------------------------------------------------------------


async def generate(
    client: client_.Client,
    model: model_.Model,
    messages: list[messages_.Message],
    params: GenerateParams,
) -> messages_.Message:
    """Generate media (images or video) through the AI Gateway.

    Dispatches to ``/image-model`` or ``/video-model`` based on ``params``
    type.
    """
    if isinstance(params, VideoParams):
        return await _generate_video(client, model, messages, params)
    return await _generate_image(client, model, messages, params)
