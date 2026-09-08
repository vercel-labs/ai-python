"""Version-independent pieces of the AI Gateway wire protocols."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx2 as httpx
import pydantic
import pydantic.alias_generators

from .... import models, ops, types
from .. import client as gateway_client
from .. import errors
from .. import params as gateway_params

# ---------------------------------------------------------------------------
# Media request helpers
# ---------------------------------------------------------------------------


def _detect_image_or_video(data: bytes | str) -> str | None:
    return types.media.detect_image_media_type(
        data
    ) or types.media.detect_media_type(data, types.media.VIDEO_SIGNATURES)


def _file_input_to_wire(
    value: types.messages.FilePart | bytes | str,
    *,
    detect: Callable[[bytes | str], str | None],
    default_media_type: str,
) -> dict[str, Any]:
    """Convert a prompt file input to the media-model input-file wire format.

    Unlike the language prompt encodings, the image- and video-model
    endpoints accept ``http(s)`` URLs directly (``{"type": "url"}``) in
    every protocol version, so they are passed through instead of
    downloaded; inline data is raw base64, not a data URL.  ``data:`` URLs
    are decoded into inline data.  The media type comes from the
    :class:`FilePart` / data URL when available, magic-byte detection via
    *detect* with *default_media_type* as the fallback otherwise.
    """
    media_type: str | None = None
    if isinstance(value, types.messages.FilePart):
        media_type = value.media_type
        value = value.data
    if isinstance(value, str):
        if types.media.is_downloadable_url(value):
            wire: dict[str, Any] = {"type": "url", "url": value}
            if media_type is not None:
                wire["mediaType"] = media_type
            return wire
        if value.startswith("data:"):
            data_url_media_type, b64 = types.media.split_data_url(value)
            media_type = media_type or data_url_media_type
            value = b64 if b64 is not None else value
    data = types.media.data_to_base64(value)
    return {
        "type": "file",
        "data": data,
        "mediaType": media_type or detect(data) or default_media_type,
    }


def parse_warnings(data: Any) -> list[ops.items.Warning]:
    """Parse the wire ``warnings`` array into :class:`~ai.ops.Warning`."""
    if not isinstance(data, list):
        return []
    return [
        ops.items.Warning(
            kind=entry.get("type") or "other",
            message=entry.get("message"),
            feature=entry.get("feature"),
            setting=entry.get("setting"),
            details=entry.get("details"),
        )
        for entry in data
        if isinstance(entry, dict)
    ]


# ---------------------------------------------------------------------------
# Tool wire format
# ---------------------------------------------------------------------------


# Free-form payload fields whose keys are data, not config structure —
# their subtrees must reach the wire verbatim, never camelized.
_OPAQUE_ARG_KEYS: dict[str, frozenset[str]] = {
    "openai.mcp": frozenset({"headers", "allowed_tools"}),
    "openai.file_search": frozenset({"filters"}),
    "openai.tool_search": frozenset({"parameters", "execution"}),
}


def _camelize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            pydantic.alias_generators.to_camel(k): _camelize(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_camelize(v) for v in value]
    return value


def tool_to_wire(tool: types.tools.Tool) -> dict[str, Any]:
    """Convert a tool schema blob to the wire format (same in all versions)."""
    if tool.kind == "provider":
        cfg = tool.tool_config
        tool_id = cfg.id if cfg is not None else None
        if tool_id is None:
            raise TypeError(
                f"provider tool {tool.name!r} has no tool_config id"
            )
        opaque = _OPAQUE_ARG_KEYS.get(tool_id, frozenset())
        return {
            "type": "provider",
            "id": tool_id,
            "name": tool.name,
            "args": {
                pydantic.alias_generators.to_camel(k): v
                if k in opaque
                else _camelize(v)
                for k, v in (cfg.args if cfg is not None else {}).items()
            },
        }
    spec = tool.spec
    if spec is None:
        raise TypeError(f"function tool {tool.name!r} has no spec")
    return {
        "type": "function",
        "name": tool.name,
        "description": spec.description or "",
        "inputSchema": spec.params,
    }


def response_format(output_type: type[Any] | None) -> dict[str, Any] | None:
    """Build the ``responseFormat`` field for a structured-output request."""
    if output_type is None or not issubclass(output_type, pydantic.BaseModel):
        return None
    return {
        "type": "json",
        "schema": output_type.model_json_schema(),
        "name": output_type.__name__,
    }


def provider_from_model_id(model_id: str) -> str | None:
    provider, sep, _ = model_id.partition("/")
    return provider if sep else None


def body_provider_options(
    body: dict[str, Any], provider: str
) -> dict[str, Any]:
    provider_options = body.setdefault("providerOptions", {})
    if not isinstance(provider_options, dict):
        raise TypeError("providerOptions must be a dict")
    options = provider_options.setdefault(provider, {})
    if not isinstance(options, dict):
        raise TypeError(f"providerOptions.{provider} must be a dict")
    return options


def merge_extra_body(
    body: dict[str, Any], extra_body: Mapping[str, Any]
) -> None:
    extra = dict(extra_body)
    provider_options = extra.pop("providerOptions", None)
    body.update(extra)
    if provider_options is None:
        return
    if not isinstance(provider_options, Mapping):
        raise TypeError("extra_body.providerOptions must be a mapping")
    existing = body.setdefault("providerOptions", {})
    if not isinstance(existing, dict):
        raise TypeError("providerOptions must be a dict")
    for provider, options in provider_options.items():
        if not isinstance(provider, str):
            raise TypeError("providerOptions keys must be strings")
        if not isinstance(options, Mapping):
            raise TypeError(f"providerOptions.{provider} must be a mapping")
        current = existing.setdefault(provider, {})
        if not isinstance(current, dict):
            raise TypeError(f"providerOptions.{provider} must be a dict")
        current.update(options)


def request_options(
    value: models.InferenceRequestParams | None,
    *,
    model_id: str,
) -> tuple[dict[str, Any], dict[str, str] | None, dict[str, Any] | None]:
    """Map request params to ``(body fields, extra headers, extra query)``.

    Applies everything except ``reasoning`` and ``extra_body``.
    """
    if value is None:
        return {}, None, None
    if not isinstance(value, models.InferenceRequestParams):
        raise TypeError(
            "ai-gateway stream params must be InferenceRequestParams"
        )

    body: dict[str, Any] = {}
    provider = provider_from_model_id(model_id)

    # routing
    if value.routing is not None:
        routing = value.routing
        options = body_provider_options(body, "gateway")
        if routing.provider_allowlist is not None:
            options["only"] = sorted(routing.provider_allowlist)
        if routing.provider_order is not None:
            options["order"] = list(routing.provider_order)
        if routing.provider_ranking is not None:
            options["sort"] = routing.provider_ranking
        if routing.fallback_models is not None:
            options["models"] = list(routing.fallback_models)
        if routing.routing_target is not None:
            target = routing.routing_target
            gateway_target = (
                target.gateway
                if isinstance(target, models.RoutingTargetChain)
                else target
            )
            if gateway_target is models.GLOBAL:
                options["inferenceRegion"] = {"scope": "global"}
            elif isinstance(gateway_target, models.GeoRegion):
                options["inferenceRegion"] = {"geoRegion": str(gateway_target)}
            else:
                options["inferenceRegion"] = {
                    "providerRegion": str(gateway_target)
                }
        if isinstance(routing.routing_target, models.RoutingTargetChain):
            leg = routing.routing_target.provider
            target_provider = provider or "gateway"
            target_options = body_provider_options(body, target_provider)
            leg_value = "global" if leg is models.GLOBAL else str(leg)
            if target_provider == "anthropic" and isinstance(
                leg, models.GeoRegion
            ):
                target_options["inferenceGeo"] = leg_value
            else:
                target_options["region"] = leg_value

    # gateway params (provider_params[GatewayParams])
    if value.provider_params is not None:
        if not isinstance(value.provider_params, Mapping):
            raise TypeError("ai-gateway provider_params must be a mapping")
        gw_params = value.provider_params.get(gateway_params.GatewayParams)
        if gw_params is not None and not isinstance(
            gw_params, gateway_params.GatewayParams
        ):
            raise TypeError(
                "ai-gateway provider_params[GatewayParams] "
                "must be GatewayParams"
            )
        if gw_params is not None:
            options = body_provider_options(body, "gateway")
            if gw_params.quota_entity_id is not None:
                options["quotaEntityId"] = gw_params.quota_entity_id
            if gw_params.zero_data_retention is not None:
                options["zeroDataRetention"] = gw_params.zero_data_retention
            if gw_params.hipaa_compliant is not None:
                options["hipaaCompliant"] = gw_params.hipaa_compliant
            if gw_params.disallow_prompt_training is not None:
                options["disallowPromptTraining"] = (
                    gw_params.disallow_prompt_training
                )
            if gw_params.byok is not None:
                options["byok"] = {
                    byok_provider: [
                        dict(credential) for credential in credentials
                    ]
                    for byok_provider, credentials in gw_params.byok.items()
                }
            if gw_params.provider_timeouts is not None:
                provider_timeouts: dict[str, Any] = {}
                if gw_params.provider_timeouts.byok is not None:
                    provider_timeouts["byok"] = dict(
                        gw_params.provider_timeouts.byok
                    )
                if provider_timeouts:
                    options["providerTimeouts"] = provider_timeouts

    # sampling
    if not isinstance(value.sampling, models.ModelProviderDefault):
        for sampler in value.sampling.values():
            match sampler:
                case models.TemperatureSamplerParams(temperature=temperature):
                    if not isinstance(temperature, models.ModelProviderDefault):
                        body["temperature"] = temperature
                case models.TopPSamplerParams(top_p=top_p):
                    if not isinstance(top_p, models.ModelProviderDefault):
                        body["topP"] = top_p
                case models.SeedSamplerParams(seed=seed):
                    if isinstance(seed, int):
                        if seed != -1:
                            body["seed"] = seed
                    elif seed is not None and not isinstance(
                        seed, models.RandomSeed | models.ModelProviderDefault
                    ):
                        raise TypeError(
                            "seed must be an int, RANDOM, DEFAULT, or None"
                        )
                case models.TopKSamplerParams(top_k=top_k):
                    if not isinstance(top_k, models.ModelProviderDefault):
                        body["topK"] = top_k
                case models.MinPSamplerParams(min_p=min_p):
                    if min_p is not None and not isinstance(
                        min_p, models.ModelProviderDefault
                    ):
                        raise ValueError("AI Gateway does not support min_p")
                case models.RepetitionPenaltyParams() as repetition:
                    if not isinstance(
                        repetition.frequency_penalty,
                        models.ModelProviderDefault,
                    ):
                        body["frequencyPenalty"] = repetition.frequency_penalty
                    if not isinstance(
                        repetition.presence_penalty,
                        models.ModelProviderDefault,
                    ):
                        body["presencePenalty"] = repetition.presence_penalty
                    if repetition.repetition_penalty is not None and (
                        not isinstance(
                            repetition.repetition_penalty,
                            models.ModelProviderDefault,
                        )
                    ):
                        raise ValueError(
                            "AI Gateway does not support repetition_penalty"
                        )
                    if repetition.consideration_window is not None and (
                        not isinstance(
                            repetition.consideration_window,
                            models.ModelProviderDefault,
                        )
                    ):
                        raise ValueError(
                            "AI Gateway does not support consideration_window"
                        )

    # context management
    context_management = value.context_management
    if (
        context_management is not None
        and context_management.compaction is not None
    ):
        threshold = context_management.compaction.value
        if provider == "openai":
            body_provider_options(body, "openai")["contextManagement"] = [
                {"type": "compaction", "compactThreshold": threshold}
            ]
        elif provider == "anthropic":
            body_provider_options(body, "anthropic")["contextManagement"] = {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": threshold,
                        },
                    }
                ]
            }
        else:
            raise ValueError(
                "AI Gateway context management requires an OpenAI or "
                "Anthropic model"
            )

    # tool calling
    if value.tool_calling is not None:
        tool_calling = value.tool_calling
        if not isinstance(
            tool_calling.max_tool_calls, models.ModelProviderDefault
        ):
            body["maxToolCalls"] = tool_calling.max_tool_calls
        if not isinstance(
            tool_calling.parallel_tool_calls, models.ModelProviderDefault
        ):
            body["parallelToolCalls"] = tool_calling.parallel_tool_calls
        tool_choice = tool_calling.tool_choice
        if isinstance(tool_choice, models.ToolChoiceMode):
            body["toolChoice"] = tool_choice.value
        elif isinstance(tool_choice, models.ToolRef):
            body["toolChoice"] = {
                "type": "tool",
                "toolName": str(tool_choice),
            }
        else:
            body["activeTools"] = sorted(
                str(tool) for tool in tool_choice.tools
            )
            body["toolChoice"] = tool_choice.mode.value

    # provider service
    if value.provider_service is not None:
        options = body_provider_options(body, provider or "gateway")
        if not isinstance(
            value.provider_service.service_tier, models.ModelProviderDefault
        ):
            options["serviceTier"] = value.provider_service.service_tier

    # safety identifier, metadata, tags
    if value.safety_identifier is not None:
        body_provider_options(body, "gateway")["user"] = value.safety_identifier

    if value.metadata is not None:
        body["metadata"] = dict(value.metadata)
        if provider in {"openai", "anthropic"}:
            body_provider_options(body, provider)["metadata"] = dict(
                value.metadata
            )

    if value.tags is not None:
        body_provider_options(body, "gateway")["tags"] = sorted(value.tags)

    # output
    if value.output is not None:
        output = value.output
        if output.max_tokens is not None:
            body["maxOutputTokens"] = output.max_tokens
        if output.include is not None:
            if provider == "openai":
                body_provider_options(body, "openai")["include"] = sorted(
                    output.include
                )
            else:
                body["include"] = sorted(output.include)
        if output.text_verbosity is not None and not isinstance(
            output.text_verbosity, models.ModelProviderDefault
        ):
            raise ValueError("AI Gateway does not support text verbosity")

    # cache
    if value.cache is not None:
        cache = value.cache
        if not isinstance(cache.mode, models.ModelProviderDefault):
            body_provider_options(body, "gateway")["caching"] = cache.mode
        if provider == "openai":
            options = body_provider_options(body, "openai")
            if cache.key is not None:
                options["promptCacheKey"] = cache.key
            if not isinstance(cache.retention, models.ModelProviderDefault):
                options["promptCacheRetention"] = cache.retention
        elif (
            not isinstance(cache.retention, models.ModelProviderDefault)
            or cache.key is not None
        ):
            options = body_provider_options(body, "gateway")
            if cache.key is not None:
                options["cacheKey"] = cache.key
            if not isinstance(cache.retention, models.ModelProviderDefault):
                options["cacheRetention"] = cache.retention

    extra_headers: dict[str, str] | None = None
    if value.extra_headers is not None:
        extra_headers = {
            key: header
            for key, header in value.extra_headers.items()
            if not isinstance(header, models.Unset)
        }
    return (
        body,
        extra_headers,
        dict(value.extra_query) if value.extra_query is not None else None,
    )


# ---------------------------------------------------------------------------
# Streaming response sub-parsers
# ---------------------------------------------------------------------------


def is_provider_executed(data: dict[str, Any]) -> bool:
    """Whether a wire tool part marks itself as provider-executed."""
    return bool(data.get("providerExecuted") or data.get("provider_executed"))


def expand_tool_call(
    data: dict[str, Any],
    streamed_tool_ids: set[str],
    provider_executed_ids: set[str] | None = None,
) -> list[types.events.Event]:
    """Expand a complete ``tool-call`` part into Start + Delta + End.

    Returns empty when the tool was already streamed via ``tool-input-*``.
    """
    tc_id = data.get("toolCallId", "")
    if tc_id in streamed_tool_ids:
        return []
    if provider_executed_ids is None:
        provider_executed_ids = set()
    tool_name = data.get("toolName", "")
    tool_input = data.get("input", "")
    args_str = (
        tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
    )
    if is_provider_executed(data) or tc_id in provider_executed_ids:
        provider_executed_ids.add(tc_id)
        return [
            types.events.BuiltinToolStart(
                tool_call_id=tc_id, tool_name=tool_name
            ),
            types.events.BuiltinToolDelta(tool_call_id=tc_id, chunk=args_str),
            types.events.BuiltinToolEnd(
                tool_call_id=tc_id,
                tool_call=types.messages.BuiltinToolCallPart(
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    tool_args=args_str,
                ),
            ),
        ]
    return [
        types.events.ToolStart(tool_call_id=tc_id, tool_name=tool_name),
        types.events.ToolDelta(tool_call_id=tc_id, chunk=args_str),
        types.events.ToolEnd(
            tool_call_id=tc_id, tool_call=types.messages.DUMMY_TOOL_CALL
        ),
    ]


# AI SDK finish reasons → the framework's finish reasons (the gen_ai
# semconv vocabulary, see ``types.events.StreamEnd``); unmapped values
# (``other``, ``unknown``) pass through raw.
FINISH_REASONS: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "content-filter": "content_filter",
    "tool-calls": "tool_call",
    "error": "error",
    "other": "other",
}


def parse_finish_reason(
    value: Any,
) -> tuple[str | None, dict[str, Any] | None]:
    """Normalize a wire finish reason.

    Accepts both shapes the gateway has used: a bare string (early v3) and
    the ``{unified, raw}`` object (late v3 and v4).  Returns the normalized
    finish reason plus provider metadata carrying the raw provider reason
    when it adds information.  ``"unknown"`` means the provider didn't
    report a reason.
    """
    raw_reason: str | None = None
    if isinstance(value, dict):
        raw = value.get("raw")
        raw_reason = raw if isinstance(raw, str) else None
        value = value.get("unified")
    finish_reason: str | None = None
    if isinstance(value, str) and value != "unknown":
        finish_reason = FINISH_REASONS.get(value, "other")
        if value not in FINISH_REASONS:
            raw_reason = raw_reason or value
    if raw_reason is not None and raw_reason != finish_reason:
        return finish_reason, {"gateway": {"finish_reason": raw_reason}}
    return finish_reason, None


def parse_usage(data: Any) -> types.usage.Usage:
    """Parse wire usage data into an internal ``Usage``."""
    if not isinstance(data, dict):
        return types.usage.Usage()

    input_tokens_obj = data.get("inputTokens")
    output_tokens_obj = data.get("outputTokens")

    if isinstance(input_tokens_obj, dict) or isinstance(
        output_tokens_obj, dict
    ):
        inp = input_tokens_obj if isinstance(input_tokens_obj, dict) else {}
        out = output_tokens_obj if isinstance(output_tokens_obj, dict) else {}
        return types.usage.Usage(
            input_tokens=inp.get("total") or 0,
            output_tokens=out.get("total") or 0,
            reasoning_tokens=out.get("reasoning"),
            cache_read_tokens=inp.get("cacheRead"),
            cache_write_tokens=inp.get("cacheWrite"),
            raw=data,
        )

    return types.usage.Usage(
        input_tokens=data.get("prompt_tokens") or data.get("inputTokens") or 0,
        output_tokens=(
            data.get("completion_tokens") or data.get("outputTokens") or 0
        ),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def _image_to_wire(
    value: types.messages.FilePart | bytes | str,
) -> dict[str, Any]:
    return _file_input_to_wire(
        value,
        detect=types.media.detect_image_media_type,
        default_media_type="image/png",
    )


async def generate_image(
    gateway: gateway_client.GatewayClient,
    model: models.Model,
    prompt: ops.images.ImagePrompt,
    *,
    params: ops.images.ImageParams,
    spec_version: str = "3",
) -> ops.items.Item[list[types.messages.FilePart]]:
    """Hit ``/image-model`` and return an Item with FileParts."""
    body: dict[str, Any] = {"n": params.n}
    if prompt.text is not None:
        body["prompt"] = prompt.text
    if params.size is not None:
        body["size"] = params.size
    if params.aspect_ratio is not None:
        body["aspectRatio"] = params.aspect_ratio
    if params.seed is not None:
        body["seed"] = params.seed
    if params.provider_options:
        body["providerOptions"] = dict(params.provider_options)
    if prompt.images:
        body["files"] = [_image_to_wire(image) for image in prompt.images]
    if prompt.mask is not None:
        body["mask"] = _image_to_wire(prompt.mask)

    try:
        response = await gateway.post(
            "image-model",
            body,
            model=model,
            model_type="image",
            spec_version=spec_version,
        )
    except gateway_client.errors.GatewayError as exc:
        raise errors.map_error(exc) from exc

    data = response.json()
    raw_images: list[str] = data.get("images", [])
    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = types.usage.Usage(
            input_tokens=usage_data.get("inputTokens") or 0,
            output_tokens=usage_data.get("outputTokens") or 0,
        )

    files: list[types.messages.FilePart] = []
    for img_b64 in raw_images:
        media_type = types.media.detect_image_media_type(img_b64) or "image/png"
        files.append(
            types.messages.FilePart(data=img_b64, media_type=media_type)
        )

    return ops.items.Item(
        value=files,
        usage=usage,
        warnings=parse_warnings(data.get("warnings")),
        provider_metadata=data.get("providerMetadata"),
    )


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


def _video_reference_to_wire(
    value: types.messages.FilePart | bytes | str,
) -> dict[str, Any]:
    return _file_input_to_wire(
        value,
        detect=_detect_image_or_video,
        default_media_type="image/png",
    )


async def generate_video(
    gateway: gateway_client.GatewayClient,
    model: models.Model,
    prompt: ops.videos.VideoPrompt,
    *,
    params: ops.videos.VideoParams,
    spec_version: str = "3",
) -> ops.items.Item[list[types.messages.FilePart]]:
    """Hit ``/video-model`` (SSE) and return an Item with FileParts.

    Normalizes the prompt into the spec slots: a ``first_frame`` frame
    image takes precedence over ``prompt.image`` as the start image, and
    frame images suppress references (the spec forbids combining them);
    both conflicts surface as warnings on the returned Item.
    """
    warnings: list[ops.items.Warning] = []
    frame_images = list(prompt.frame_images)
    references = list(prompt.references)
    if frame_images and references:
        warnings.append(
            ops.items.Warning(
                message="references were ignored because frame_images "
                "were provided; frame_images and references cannot be "
                "combined."
            )
        )
        references = []
    first_frame = next(
        (f.image for f in frame_images if f.frame_type == "first_frame"),
        None,
    )
    image = prompt.image
    if image is not None and first_frame is not None:
        warnings.append(
            ops.items.Warning(
                message="image was ignored because a first_frame frame "
                "image was provided; the first_frame image takes "
                "precedence as the start image."
            )
        )
    start_image = first_frame if first_frame is not None else image

    body: dict[str, Any] = {"n": params.n}
    if prompt.text is not None:
        body["prompt"] = prompt.text
    if params.aspect_ratio is not None:
        body["aspectRatio"] = params.aspect_ratio
    if params.resolution is not None:
        body["resolution"] = params.resolution
    if params.duration is not None:
        body["duration"] = params.duration
    if params.fps is not None:
        body["fps"] = params.fps
    if params.seed is not None:
        body["seed"] = params.seed
    if params.generate_audio is not None:
        body["generateAudio"] = params.generate_audio
    if params.provider_options:
        body["providerOptions"] = dict(params.provider_options)
    if start_image is not None:
        body["image"] = _image_to_wire(start_image)
    if frame_images:
        body["frameImages"] = [
            {
                "image": _image_to_wire(frame.image),
                "frameType": frame.frame_type,
            }
            for frame in frame_images
        ]
    if references:
        body["inputReferences"] = [
            _video_reference_to_wire(reference) for reference in references
        ]

    try:
        async with gateway.stream(
            "video-model",
            body,
            model=model,
            model_type="video",
            accept="text/event-stream",
            timeout=httpx.Timeout(timeout=600.0, connect=10.0),
            spec_version=spec_version,
        ) as response:
            event_data: dict[str, Any] = {}
            async for parsed in gateway.iter_sse(response):
                event_data = parsed
                break

        if not event_data:
            raise gateway_client.errors.GatewayResponseError(
                "SSE stream ended without any data events",
            )

        if event_data.get("type") == "error":
            raise gateway_client.errors.GatewayInvalidRequestError(
                message=event_data.get("message", "unknown error"),
                status_code=event_data.get("statusCode", 400),
            )
    except gateway_client.errors.GatewayError as exc:
        raise errors.map_error(exc) from exc

    raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
    files: list[types.messages.FilePart] = []
    for video_data in raw_videos:
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            (
                downloaded_bytes,
                content_type,
            ) = await models.core.helpers.files.download(video_data["url"])
            if content_type:
                media_type = content_type
            files.append(
                types.messages.FilePart(
                    data=downloaded_bytes, media_type=media_type
                )
            )
        else:
            raw_data = video_data.get("data", "")
            files.append(
                types.messages.FilePart(data=raw_data, media_type=media_type)
            )

    return ops.items.Item(
        value=files,
        warnings=warnings + parse_warnings(event_data.get("warnings")),
        provider_metadata=event_data.get("providerMetadata"),
    )


# ---------------------------------------------------------------------------
# Speech generation
# ---------------------------------------------------------------------------


async def generate_audio(
    gateway: gateway_client.GatewayClient,
    model: models.Model,
    prompt: ops.audio.AudioPrompt,
    *,
    params: ops.audio.AudioParams,
    spec_version: str = "3",
) -> ops.items.Item[list[types.messages.FilePart]]:
    """Hit ``/speech-model`` and return an Item with a FilePart."""
    body: dict[str, Any] = {
        "text": prompt.text,
    }
    if prompt.instructions is not None:
        body["instructions"] = prompt.instructions
    if params.voice is not None:
        body["voice"] = params.voice
    if params.output_format is not None:
        body["outputFormat"] = params.output_format
    if params.speed is not None:
        body["speed"] = params.speed
    if params.language is not None:
        body["language"] = params.language
    if params.provider_options:
        body["providerOptions"] = dict(params.provider_options)

    try:
        response = await gateway.post(
            "speech-model",
            body,
            model=model,
            model_type="speech",
            spec_version=spec_version,
        )
    except gateway_client.errors.GatewayError as exc:
        raise errors.map_error(exc) from exc

    data = response.json()
    audio_b64: str = data.get("audio") or ""

    files: list[types.messages.FilePart] = []
    if audio_b64:
        media_type = (
            types.media.detect_audio_media_type(audio_b64) or "audio/mpeg"
        )
        files.append(
            types.messages.FilePart(data=audio_b64, media_type=media_type)
        )

    return ops.items.Item(
        value=files,
        warnings=parse_warnings(data.get("warnings")),
        provider_metadata=data.get("providerMetadata"),
    )
