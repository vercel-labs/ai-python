"""AI Gateway v3 protocol.

Converts internal messages to AI Gateway wire payloads and maps gateway
responses back to public event/message types.
"""

import base64
import json
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from typing import Any, TypeVar

import httpx
import pydantic

from ... import types
from ...models import core
from ...models.core import params as params_
from .. import base
from ..anthropic import tools as anthropic_tools
from ..openai import tools as openai_tools
from . import client as gateway_client
from . import errors
from . import params as gateway_params
from . import tools as gateway_tools
from .client import errors as client_errors

# ---------------------------------------------------------------------------
# Shared request helpers
# ---------------------------------------------------------------------------


_ProviderParamsT = TypeVar("_ProviderParamsT")


def _provider_params_value(
    value: Mapping[type[Any], Any] | None,
    params_type: type[_ProviderParamsT],
    *,
    provider: str,
) -> _ProviderParamsT | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{provider} provider_params must be a mapping")
    provider_params = value.get(params_type)
    if provider_params is None:
        return None
    if not isinstance(provider_params, params_type):
        raise TypeError(
            f"{provider} provider_params[{params_type.__name__}] "
            f"must be {params_type.__name__}"
        )
    return provider_params


def _extract_prompt(messages: list[types.messages.Message]) -> str:
    """Concatenate all text from user/system messages into one prompt."""
    parts: list[str] = []
    for msg in messages:
        if msg.role in ("user", "system"):
            for p in msg.parts:
                if isinstance(p, types.messages.TextPart):
                    parts.append(p.text)
    return " ".join(parts)


def _extract_input_files(
    messages: list[types.messages.Message],
) -> list[types.messages.FilePart]:
    """Collect all file parts from user messages."""
    files_: list[types.messages.FilePart] = []
    for msg in messages:
        if msg.role == "user":
            for p in msg.parts:
                if isinstance(p, types.messages.FilePart):
                    files_.append(p)
    return files_


def _file_part_to_wire(part: types.messages.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to gateway input-file wire format."""
    data = part.data
    if isinstance(data, str) and types.media.is_url(data):
        return {"type": "url", "url": data}
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    elif isinstance(data, str):
        b64 = data
    else:
        b64 = str(data)
    return {"type": "file", "data": b64, "mediaType": part.media_type}


# ---------------------------------------------------------------------------
# Tool result output -> v3 wire
# ---------------------------------------------------------------------------


def _file_part_to_v3_inline(part: types.messages.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to an inline v3 content element.

    Images become ``image-data``; everything else becomes ``file-data``.
    """
    b64 = types.media.data_to_base64(part.data)
    if part.media_type.startswith("image/"):
        return {"type": "image-data", "data": b64, "mediaType": part.media_type}
    entry: dict[str, Any] = {
        "type": "file-data",
        "data": b64,
        "mediaType": part.media_type,
    }
    if part.filename is not None:
        entry["filename"] = part.filename
    return entry


def _tool_result_output(
    part: types.messages.ToolResultPart,
) -> dict[str, Any]:
    """Convert a tool result to its v3 ``output`` wire form.

    The v3 protocol carries a tagged output union.  A :class:`ContentOutput`
    becomes ``content``; an error result becomes ``error-text`` (for a
    ``str``) or ``error-json``; otherwise ``text`` (for a ``str``) or
    ``json``.  The text-vs-json call is made here, at the wire boundary.
    """
    value = part.get_model_input()
    if isinstance(value, types.messages.ContentOutput):
        parts: list[dict[str, Any]] = []
        for item in value.value:
            if isinstance(item, types.messages.FilePart):
                parts.append(_file_part_to_v3_inline(item))
            else:
                parts.append({"type": "text", "text": item.text})
        return {"type": "content", "value": parts}
    if part.is_error:
        if value is None or isinstance(value, str):
            return {"type": "error-text", "value": value or ""}
        return {"type": "error-json", "value": value}
    if value is None or isinstance(value, str):
        return {"type": "text", "value": value or ""}
    return {"type": "json", "value": value}


# ---------------------------------------------------------------------------
# Streaming request building — Message list → v3 prompt
# ---------------------------------------------------------------------------


async def _file_part_to_v3(part: types.messages.FilePart) -> dict[str, Any]:
    """Convert a :class:`FilePart` to a v3 ``file`` content part."""
    data = part.data
    if isinstance(data, str) and types.media.is_downloadable_url(data):
        downloaded, _ = await core.helpers.files.download(data)
        data = downloaded

    entry: dict[str, Any] = {
        "type": "file",
        "mediaType": part.media_type,
        "data": types.media.data_to_data_url(data, part.media_type),
    }
    if part.filename is not None:
        entry["filename"] = part.filename
    return entry


async def _messages_to_prompt(
    messages: list[types.messages.Message],
) -> list[dict[str, Any]]:
    """Convert ``Message`` list to the v3 prompt wire format."""
    result: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                text = "".join(
                    p.text
                    for p in msg.parts
                    if isinstance(p, types.messages.TextPart)
                )
                result.append({"role": "system", "content": text})

            case "user":
                content: list[dict[str, Any]] = []
                for p in msg.parts:
                    if isinstance(p, types.messages.TextPart):
                        content.append({"type": "text", "text": p.text})
                    elif isinstance(p, types.messages.FilePart):
                        content.append(await _file_part_to_v3(p))
                result.append({"role": "user", "content": content})

            case "assistant":
                assistant_content: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.messages.ReasoningPart(
                            text=text, provider_metadata=pm
                        ):
                            reasoning_entry: dict[str, Any] = {
                                "type": "reasoning",
                                "text": text,
                            }
                            # Replay the provider's reasoning metadata (e.g.
                            # the thinking-block signature) verbatim. Without
                            # it the provider drops the block and the model
                            # loses access to its prior reasoning. v3 mirrors
                            # inbound ``providerMetadata`` to outbound
                            # ``providerOptions``.
                            if pm:
                                reasoning_entry["providerOptions"] = pm
                            assistant_content.append(reasoning_entry)
                        case types.messages.TextPart(text=text):
                            assistant_content.append(
                                {"type": "text", "text": text}
                            )
                        case types.messages.ToolCallPart() as tp:
                            tool_input: Any = (
                                json.loads(tp.tool_args) if tp.tool_args else {}
                            )
                            assistant_content.append(
                                {
                                    "type": "tool-call",
                                    "toolCallId": tp.tool_call_id,
                                    "toolName": tp.tool_name,
                                    "input": tool_input,
                                }
                            )
                        case types.messages.BuiltinToolCallPart() as btp:
                            btp_input: Any = (
                                json.loads(btp.tool_args)
                                if btp.tool_args
                                else {}
                            )
                            assistant_content.append(
                                {
                                    "type": "tool-call",
                                    "toolCallId": btp.tool_call_id,
                                    "toolName": btp.tool_name,
                                    "input": btp_input,
                                    "providerExecuted": True,
                                }
                            )
                        case types.messages.BuiltinToolReturnPart() as brp:
                            assistant_content.append(
                                {
                                    "type": "tool-result",
                                    "toolCallId": brp.tool_call_id,
                                    "toolName": brp.tool_name,
                                    "output": {
                                        "type": "json",
                                        "value": brp.result,
                                    },
                                    "providerExecuted": True,
                                }
                            )
                result.append(
                    {"role": "assistant", "content": assistant_content}
                )

            case "tool":
                tool_results: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, types.messages.ToolResultPart):
                        output = _tool_result_output(part)
                        tool_results.append(
                            {
                                "type": "tool-result",
                                "toolCallId": part.tool_call_id,
                                "toolName": part.tool_name,
                                "output": output,
                            }
                        )
                if tool_results:
                    result.append({"role": "tool", "content": tool_results})

    return result


def _tool_to_v3(tool: types.tools.Tool) -> dict[str, Any]:
    """Convert a tool schema blob to the v3 wire format."""
    if tool.kind == "provider":
        return {
            "type": "provider",
            "id": _provider_tool_id(tool),
            "name": tool.name,
            "args": tool.args.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            ),
        }
    args = tool.args
    if not isinstance(args, types.tools.FunctionToolArgs):
        raise TypeError(f"function tool {tool.name!r} has invalid args")
    return {
        "type": "function",
        "name": tool.name,
        "description": args.description or "",
        "inputSchema": args.params,
    }


def _provider_tool_id(tool: types.tools.Tool) -> str:
    if isinstance(tool.args, anthropic_tools.AnthropicProviderArgs):
        return f"anthropic.{tool.args.anthropic_type}"
    if isinstance(tool.args, openai_tools.OpenAIProviderArgs):
        return tool.args.openai_id

    match tool.args:
        case gateway_tools.PerplexitySearchArgs():
            return "gateway.perplexity_search"
        case gateway_tools.ParallelSearchArgs():
            return "gateway.parallel_search"
        case _:
            raise TypeError(
                f"provider tool {tool.name!r} has unsupported args "
                f"{type(tool.args).__name__}"
            )


async def _build_request_body(
    messages: list[types.messages.Message],
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``LanguageModelV3CallOptions`` request body."""
    body: dict[str, Any] = dict(params or {})
    body["prompt"] = await _messages_to_prompt(messages)
    if tools:
        body["tools"] = [_tool_to_v3(tool) for tool in tools]
    if output_type is not None and issubclass(output_type, pydantic.BaseModel):
        body["responseFormat"] = {
            "type": "json",
            "schema": output_type.model_json_schema(),
            "name": output_type.__name__,
        }
    return body


def _is_default(value: object) -> bool:
    return isinstance(value, params_.ModelProviderDefault)


def _not_default(value: object) -> bool:
    return not _is_default(value)


def _seed_value(seed: object) -> int | None:
    if seed is None or seed == -1 or isinstance(seed, params_.RandomSeed):
        return None
    if isinstance(seed, int):
        return seed
    if isinstance(seed, params_.ModelProviderDefault):
        return None
    raise TypeError("seed must be an int, RANDOM, DEFAULT, or None")


def _filter_extra_headers(
    headers: Mapping[str, str | params_.Unset] | None,
) -> dict[str, str] | None:
    if headers is None:
        return None
    return {
        key: value
        for key, value in headers.items()
        if not isinstance(value, params_.Unset)
    }


def _provider_from_model_id(model_id: str) -> str | None:
    provider, sep, _ = model_id.partition("/")
    return provider if sep else None


def _body_provider_options(
    body: dict[str, Any], provider: str
) -> dict[str, Any]:
    provider_options = body.setdefault("providerOptions", {})
    if not isinstance(provider_options, dict):
        raise TypeError("providerOptions must be a dict")
    options = provider_options.setdefault(provider, {})
    if not isinstance(options, dict):
        raise TypeError(f"providerOptions.{provider} must be a dict")
    return options


def _sequence(value: Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    return list(value)


def _target_to_inference_region(
    target: params_.RoutingTarget,
) -> dict[str, str]:
    if target is params_.GLOBAL:
        return {"scope": "global"}
    if isinstance(target, params_.GeoRegion):
        return {"geoRegion": str(target)}
    return {"providerRegion": str(target)}


def _apply_provider_target(
    body: dict[str, Any],
    *,
    provider: str | None,
    target: params_.RoutingTarget,
) -> None:
    target_provider = provider or "gateway"
    options = _body_provider_options(body, target_provider)
    target_value = "global" if target is params_.GLOBAL else str(target)
    if target_provider == "anthropic" and isinstance(target, params_.GeoRegion):
        options["inferenceGeo"] = target_value
    else:
        options["region"] = target_value


def _routing_to_gateway_options(
    routing: params_.RoutingParams,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key, value in {
        "only": sorted(routing.provider_allowlist)
        if routing.provider_allowlist is not None
        else None,
        "order": _sequence(routing.provider_order),
        "sort": routing.provider_ranking,
        "models": _sequence(routing.fallback_models),
    }.items():
        if value is not None:
            options[key] = value

    if routing.routing_target is not None:
        target = routing.routing_target
        gateway_target = (
            target.gateway
            if isinstance(target, params_.RoutingTargetChain)
            else target
        )
        options["inferenceRegion"] = _target_to_inference_region(gateway_target)

    return options


def _apply_gateway_routing(
    body: dict[str, Any],
    routing: params_.RoutingParams | None,
    *,
    provider: str | None,
) -> None:
    if routing is None:
        return
    _body_provider_options(body, "gateway").update(
        _routing_to_gateway_options(routing)
    )
    if isinstance(routing.routing_target, params_.RoutingTargetChain):
        _apply_provider_target(
            body,
            provider=provider,
            target=routing.routing_target.provider,
        )


def _apply_gateway_params(
    body: dict[str, Any], value: gateway_params.GatewayParams | None
) -> None:
    if value is None:
        return
    options = _body_provider_options(body, "gateway")
    for key, option in {
        "quotaEntityId": value.quota_entity_id,
        "zeroDataRetention": value.zero_data_retention,
        "hipaaCompliant": value.hipaa_compliant,
        "disallowPromptTraining": value.disallow_prompt_training,
    }.items():
        if option is not None:
            options[key] = option

    if value.byok is not None:
        options["byok"] = {
            provider: [dict(credential) for credential in credentials]
            for provider, credentials in value.byok.items()
        }

    if value.provider_timeouts is not None:
        provider_timeouts: dict[str, Any] = {}
        if value.provider_timeouts.byok is not None:
            provider_timeouts["byok"] = dict(value.provider_timeouts.byok)
        if provider_timeouts:
            options["providerTimeouts"] = provider_timeouts


def _merge_extra_body(
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


def _gateway_tool_choice(
    tool_choice: params_.ToolChoiceMode
    | params_.ToolRef
    | params_.ToolSelection,
    body: dict[str, Any],
) -> str | dict[str, str]:
    if isinstance(tool_choice, params_.ToolChoiceMode):
        return tool_choice.value
    if isinstance(tool_choice, params_.ToolRef):
        return {"type": "tool", "toolName": str(tool_choice)}
    body["activeTools"] = sorted(str(tool) for tool in tool_choice.tools)
    return tool_choice.mode.value


def _apply_gateway_reasoning(
    body: dict[str, Any],
    request_params: params_.InferenceRequestParams,
    *,
    provider: str | None,
) -> None:
    reasoning = request_params.reasoning
    output = request_params.output
    effort: str | params_.ModelProviderDefault | None = params_.DEFAULT
    if not isinstance(reasoning, params_.ModelProviderDefault):
        effort = reasoning.effort
    summary = params_.DEFAULT if output is None else output.reasoning_summary
    if _is_default(effort) and _is_default(summary):
        return
    if provider == "openai":
        options = _body_provider_options(body, "openai")
        if _not_default(effort):
            options["reasoningEffort"] = effort
        if _not_default(summary):
            options["reasoningSummary"] = summary
        return
    if provider == "anthropic":
        options = _body_provider_options(body, "anthropic")
        if _not_default(effort):
            if effort is None:
                options["thinking"] = {"type": "disabled"}
            else:
                options["effort"] = effort
                # The gateway only turns thinking on when a `thinking`
                # block is present; `effort` alone is a no-op upstream.
                thinking = dict(options.get("thinking") or {})
                thinking.setdefault("type", "adaptive")
                options["thinking"] = thinking
        if _not_default(summary):
            # `reasoning_summary` only controls whether the reasoning summary
            # is surfaced; it never turns thinking off (use
            # `reasoning.effort=None` for that). `None` maps to
            # `display="omitted"` -- think, but don't emit a summary -- and is
            # ignored when thinking is already disabled.
            thinking = dict(options.get("thinking") or {})
            if thinking.get("type") != "disabled":
                thinking.setdefault("type", "adaptive")
                thinking["display"] = "omitted" if summary is None else summary
                options["thinking"] = thinking
        return
    body["reasoning"] = {
        key: value
        for key, value in {
            "effort": effort,
            "summary": summary,
        }.items()
        if _not_default(value)
    }


def _apply_gateway_context_management(
    body: dict[str, Any],
    request_params: params_.InferenceRequestParams,
    *,
    provider: str | None,
) -> None:
    context_management = request_params.context_management
    if context_management is None or context_management.compaction is None:
        return
    threshold = context_management.compaction.value
    if provider == "openai":
        _body_provider_options(body, "openai")["contextManagement"] = [
            {"type": "compaction", "compactThreshold": threshold}
        ]
        return
    if provider == "anthropic":
        _body_provider_options(body, "anthropic")["contextManagement"] = {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": threshold},
                }
            ]
        }
        return
    raise ValueError(
        "AI Gateway context management requires an OpenAI or Anthropic model"
    )


def _apply_gateway_sampling(
    body: dict[str, Any],
    request_params: params_.InferenceRequestParams,
) -> None:
    sampling = request_params.sampling
    if isinstance(sampling, params_.ModelProviderDefault):
        return
    for sampler in sampling.values():
        match sampler:
            case params_.TemperatureSamplerParams(temperature=temperature):
                if _not_default(temperature):
                    body["temperature"] = temperature
            case params_.TopPSamplerParams(top_p=top_p):
                if _not_default(top_p):
                    body["topP"] = top_p
            case params_.SeedSamplerParams(seed=seed):
                if _not_default(seed):
                    value = _seed_value(seed)
                    if value is not None:
                        body["seed"] = value
            case params_.TopKSamplerParams(top_k=top_k):
                if _not_default(top_k):
                    body["topK"] = top_k
            case params_.MinPSamplerParams(min_p=min_p):
                if _not_default(min_p) and min_p is not None:
                    raise ValueError("AI Gateway does not support min_p")
            case params_.RepetitionPenaltyParams() as repetition:
                if _not_default(repetition.frequency_penalty):
                    body["frequencyPenalty"] = repetition.frequency_penalty
                if _not_default(repetition.presence_penalty):
                    body["presencePenalty"] = repetition.presence_penalty
                if (
                    _not_default(repetition.repetition_penalty)
                    and repetition.repetition_penalty is not None
                ):
                    raise ValueError(
                        "AI Gateway does not support repetition_penalty"
                    )
                if (
                    _not_default(repetition.consideration_window)
                    and repetition.consideration_window is not None
                ):
                    raise ValueError(
                        "AI Gateway does not support consideration_window"
                    )


def _gateway_request_options(
    value: params_.InferenceRequestParams | None,
    *,
    model_id: str,
) -> tuple[dict[str, Any], dict[str, str] | None, dict[str, Any] | None]:
    if value is None:
        return {}, None, None
    if not isinstance(value, params_.InferenceRequestParams):
        raise TypeError(
            "ai-gateway stream params must be InferenceRequestParams"
        )

    body: dict[str, Any] = {}
    provider = _provider_from_model_id(model_id)
    _apply_gateway_routing(body, value.routing, provider=provider)
    _apply_gateway_params(
        body,
        _provider_params_value(
            value.provider_params,
            gateway_params.GatewayParams,
            provider="ai-gateway",
        ),
    )
    _apply_gateway_sampling(body, value)
    _apply_gateway_reasoning(body, value, provider=provider)
    _apply_gateway_context_management(body, value, provider=provider)

    if value.tool_calling is not None:
        tool_calling = value.tool_calling
        if _not_default(tool_calling.max_tool_calls):
            body["maxToolCalls"] = tool_calling.max_tool_calls
        if _not_default(tool_calling.parallel_tool_calls):
            body["parallelToolCalls"] = tool_calling.parallel_tool_calls
        body["toolChoice"] = _gateway_tool_choice(
            tool_calling.tool_choice,
            body,
        )

    if value.provider_service is not None:
        service = value.provider_service
        target_provider = provider or "gateway"
        options = _body_provider_options(body, target_provider)
        if _not_default(service.service_tier):
            options["serviceTier"] = service.service_tier

    if value.safety_identifier is not None:
        _body_provider_options(body, "gateway")["user"] = (
            value.safety_identifier
        )

    if value.metadata is not None:
        body["metadata"] = dict(value.metadata)
        if provider in {"openai", "anthropic"}:
            _body_provider_options(body, provider)["metadata"] = dict(
                value.metadata
            )

    if value.tags is not None:
        _body_provider_options(body, "gateway")["tags"] = sorted(value.tags)

    if value.output is not None:
        output = value.output
        if output.max_tokens is not None:
            body["maxOutputTokens"] = output.max_tokens
        if output.include is not None:
            if provider == "openai":
                _body_provider_options(body, "openai")["include"] = sorted(
                    output.include
                )
            else:
                body["include"] = sorted(output.include)
        if (
            _not_default(output.text_verbosity)
            and output.text_verbosity is not None
        ):
            raise ValueError("AI Gateway does not support text verbosity")

    if value.cache is not None:
        cache = value.cache
        if _not_default(cache.mode):
            _body_provider_options(body, "gateway")["caching"] = cache.mode
        if provider == "openai":
            options = _body_provider_options(body, "openai")
            if cache.key is not None:
                options["promptCacheKey"] = cache.key
            if _not_default(cache.retention):
                options["promptCacheRetention"] = cache.retention
        elif _not_default(cache.retention) or cache.key is not None:
            options = _body_provider_options(body, "gateway")
            if cache.key is not None:
                options["cacheKey"] = cache.key
            if _not_default(cache.retention):
                options["cacheRetention"] = cache.retention

    if value.extra_body is not None:
        _merge_extra_body(body, value.extra_body)

    return (
        body,
        _filter_extra_headers(value.extra_headers),
        dict(value.extra_query) if value.extra_query is not None else None,
    )


# ---------------------------------------------------------------------------
# Streaming response parsing — v3 stream parts → public Event
# ---------------------------------------------------------------------------


def _is_provider_executed(data: dict[str, Any]) -> bool:
    """Whether a v3 tool part marks itself as provider-executed."""
    return bool(data.get("providerExecuted") or data.get("provider_executed"))


def _expand_tool_call(
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
    if _is_provider_executed(data) or tc_id in provider_executed_ids:
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


def _parse_usage(data: Any) -> types.usage.Usage:
    """Parse v3 usage data into an internal ``Usage``."""
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


def _parse_stream_part(
    data: dict[str, Any],
    streamed_tool_ids: set[str],
    provider_executed_ids: set[str] | None = None,
) -> list[types.events.Event]:
    """Convert a ``LanguageModelV3StreamPart`` to public events."""
    if provider_executed_ids is None:
        provider_executed_ids = set()
    match data.get("type", ""):
        case "text-start":
            return [types.events.TextStart(block_id=data.get("id", "text"))]

        case "text-delta":
            return [
                types.events.TextDelta(
                    block_id=data.get("id", "text"),
                    chunk=data.get("textDelta", data.get("delta", "")),
                )
            ]

        case "text-end":
            return [types.events.TextEnd(block_id=data.get("id", "text"))]

        case "reasoning-start":
            # Metadata on -start is gateway routing info (generationId),
            # not the provider's reasoning metadata; don't replay it.
            return [
                types.events.ReasoningStart(
                    block_id=data.get("id", "reasoning")
                )
            ]

        case "reasoning-delta":
            return [
                types.events.ReasoningDelta(
                    block_id=data.get("id", "reasoning"),
                    chunk=data.get("delta", ""),
                    provider_metadata=data.get("providerMetadata"),
                )
            ]

        case "reasoning-end":
            return [
                types.events.ReasoningEnd(
                    block_id=data.get("id", "reasoning"),
                    provider_metadata=data.get("providerMetadata"),
                )
            ]

        case "tool-input-start":
            tcid = data.get("id", "")
            streamed_tool_ids.add(tcid)
            if _is_provider_executed(data):
                provider_executed_ids.add(tcid)
                return [
                    types.events.BuiltinToolStart(
                        tool_call_id=tcid,
                        tool_name=data.get("toolName", ""),
                    )
                ]
            return [
                types.events.ToolStart(
                    tool_call_id=tcid,
                    tool_name=data.get("toolName", ""),
                )
            ]

        case "tool-input-delta":
            tcid = data.get("id", "")
            if tcid in provider_executed_ids:
                return [
                    types.events.BuiltinToolDelta(
                        tool_call_id=tcid,
                        chunk=data.get("delta", ""),
                    )
                ]
            return [
                types.events.ToolDelta(
                    tool_call_id=tcid,
                    chunk=data.get("delta", ""),
                )
            ]

        case "tool-input-end":
            tcid = data.get("id", "")
            if tcid in provider_executed_ids:
                return [
                    types.events.BuiltinToolEnd(
                        tool_call_id=tcid,
                        tool_call=types.messages.BuiltinToolCallPart(
                            tool_call_id=tcid,
                            tool_name="",
                        ),
                    )
                ]
            return [
                types.events.ToolEnd(
                    tool_call_id=tcid,
                    tool_call=types.messages.DUMMY_TOOL_CALL,
                )
            ]

        case "tool-call":
            return _expand_tool_call(
                data, streamed_tool_ids, provider_executed_ids
            )

        case "tool-result":
            tcid = data.get("toolCallId", "")
            tool_name = data.get("toolName", "")
            output = data.get("output") or data.get("result")
            is_error = bool(data.get("isError"))
            if _is_provider_executed(data) or tcid in provider_executed_ids:
                provider_executed_ids.add(tcid)
                return [
                    types.events.BuiltinToolResult(
                        tool_call_id=tcid,
                        result=types.messages.BuiltinToolReturnPart(
                            tool_call_id=tcid,
                            tool_name=tool_name,
                            result=output,
                            is_error=is_error,
                        ),
                    )
                ]
            return []

        case "file":
            return [
                types.events.FileEvent(
                    block_id=data.get("id", ""),
                    media_type=data.get(
                        "mediaType", "application/octet-stream"
                    ),
                    data=data.get("data", ""),
                )
            ]

        case "finish":
            usage_data = data.get("usage")
            usage = _parse_usage(usage_data) if usage_data else None
            return [types.events.StreamEnd(usage=usage)]

        case _:
            return []


async def stream(
    gateway: gateway_client.GatewayClient,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: params_.InferenceRequestParams | None = None,
) -> AsyncGenerator[types.events.Event]:
    """Stream an LLM response through the AI Gateway v3 protocol."""
    stream_params, extra_headers, extra_query = _gateway_request_options(
        params,
        model_id=model.id,
    )
    body = await _build_request_body(
        messages,
        tools=tools,
        output_type=output_type,
        params=stream_params,
    )

    try:
        async with gateway.stream(
            "language-model",
            body,
            model=model,
            model_type="language",
            streaming=True,
            headers=extra_headers,
            query=extra_query,
        ) as response:
            yield types.events.StreamStart()
            streamed_tool_ids: set[str] = set()
            provider_executed_ids: set[str] = set()
            async for data in gateway.iter_sse(response):
                for event in _parse_stream_part(
                    data, streamed_tool_ids, provider_executed_ids
                ):
                    yield event
    except client_errors.GatewayError as exc:
        raise errors.map_error(exc) from exc
    except httpx.TimeoutException as exc:
        timeout_error = client_errors.GatewayTimeoutError()
        raise errors.map_error(timeout_error) from exc
    except Exception as exc:
        response_error = client_errors.GatewayResponseError(
            message=f"Unexpected error during streaming: {exc}",
        )
        raise errors.map_error(response_error) from exc


# ---------------------------------------------------------------------------
# Media generation
# ---------------------------------------------------------------------------


async def _generate_image(
    gateway: gateway_client.GatewayClient,
    model: core.model.Model,
    messages: list[types.messages.Message],
    params: core.ImageParams,
) -> types.messages.Message:
    """Hit ``/image-model`` and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["files"] = [_file_part_to_wire(f) for f in input_files]

    response = await gateway.post_json(
        "image-model", body, model=model, model_type="image"
    )

    data = response.json()
    raw_images: list[str] = data.get("images", [])
    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = types.usage.Usage(
            input_tokens=usage_data.get("inputTokens") or 0,
            output_tokens=usage_data.get("outputTokens") or 0,
        )

    parts: list[types.messages.Part] = []
    for img_b64 in raw_images:
        media_type = types.media.detect_image_media_type(img_b64) or "image/png"
        parts.append(
            types.messages.FilePart(data=img_b64, media_type=media_type)
        )

    return types.messages.Message(role="assistant", parts=parts, usage=usage)


async def _generate_video(
    gateway: gateway_client.GatewayClient,
    model: core.model.Model,
    messages: list[types.messages.Message],
    params: core.VideoParams,
) -> types.messages.Message:
    """Hit ``/video-model`` (SSE) and return a Message with FileParts."""
    prompt = _extract_prompt(messages)
    input_files = _extract_input_files(messages)

    body: dict[str, Any] = {
        "prompt": prompt,
        **params.model_dump(by_alias=True, exclude_none=True),
    }
    if input_files:
        body["image"] = _file_part_to_wire(input_files[0])

    async with gateway.stream(
        "video-model",
        body,
        model=model,
        model_type="video",
        accept="text/event-stream",
        timeout=httpx.Timeout(timeout=600.0, connect=10.0),
    ) as response:
        event_data: dict[str, Any] = {}
        async for parsed in gateway.iter_sse(response):
            event_data = parsed
            break

    if not event_data:
        raise client_errors.GatewayResponseError(
            "SSE stream ended without any data events",
        )

    if event_data.get("type") == "error":
        raise client_errors.GatewayInvalidRequestError(
            message=event_data.get("message", "unknown error"),
            status_code=event_data.get("statusCode", 400),
        )

    raw_videos: list[dict[str, Any]] = event_data.get("videos", [])
    parts: list[types.messages.Part] = []
    for video_data in raw_videos:
        vtype = video_data.get("type", "base64")
        media_type = video_data.get("mediaType", "video/mp4")

        if vtype == "url":
            downloaded_bytes, content_type = await core.helpers.files.download(
                video_data["url"]
            )
            if content_type:
                media_type = content_type
            parts.append(
                types.messages.FilePart(
                    data=downloaded_bytes, media_type=media_type
                )
            )
        else:
            raw_data = video_data.get("data", "")
            parts.append(
                types.messages.FilePart(data=raw_data, media_type=media_type)
            )

    return types.messages.Message(role="assistant", parts=parts)


async def generate(
    gateway: gateway_client.GatewayClient,
    model: core.model.Model,
    messages: list[types.messages.Message],
    params: core.GenerateParams,
) -> types.messages.Message:
    """Generate media through the AI Gateway."""
    try:
        if isinstance(params, core.VideoParams):
            return await _generate_video(gateway, model, messages, params)
        return await _generate_image(gateway, model, messages, params)
    except client_errors.GatewayError as exc:
        raise errors.map_error(exc) from exc


class GatewayV3Protocol(base.ProviderProtocol[gateway_client.GatewayClient]):
    """AI Gateway v3 wire protocol."""

    def stream(
        self,
        client: gateway_client.GatewayClient,
        model: core.model.Model,
        messages: list[types.messages.Message],
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[types.events.Event]:
        _ = provider
        return stream(
            client,
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
        )

    async def generate(
        self,
        client: gateway_client.GatewayClient,
        model: core.model.Model,
        messages: list[types.messages.Message],
        params: core.GenerateParams,
        *,
        provider: str,
    ) -> types.messages.Message:
        _ = provider
        return await generate(client, model, messages, params)
