"""Anthropic protocol — messages API.

Message/tool conversion and streaming via the official ``anthropic`` SDK.
Anthropic-compatible providers own the SDK client used by this protocol.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any, cast

import pydantic

from ... import types
from ...models.core import params as params_
from ...types import events
from .. import base
from . import _sdk, errors
from . import tools as anthropic_tools

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping, Sequence

    import anthropic

    from ...models import core

PROVIDER_NAME = "anthropic"

# Anthropic block types that carry server-tool results. We track these
# so multi-turn message mapping can round-trip them back to the API.
_TOOL_RESULT_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_tool_result",
        "memory_tool_result",
    }
)


def _provider_metadata(**values: Any) -> dict[str, Any]:
    """Namespace metadata as ``{"anthropic": {...}}``."""
    return {PROVIDER_NAME: {**values}}


def _anthropic_metadata(pm: dict[str, Any] | None) -> dict[str, Any]:
    """Read back the metadata written by :func:`_provider_metadata`."""
    meta = (pm or {}).get(PROVIDER_NAME)
    return meta if isinstance(meta, dict) else {}


# ---------------------------------------------------------------------------
# Message / tool conversion — internal types → Anthropic wire format
# ---------------------------------------------------------------------------


def _split_tools(
    tools: Sequence[types.tools.Tool],
) -> tuple[list[types.tools.Tool], list[types.tools.Tool]]:
    """Split ``tools`` into host-executed and provider-executed declarations."""
    custom: list[types.tools.Tool] = []
    builtin: list[types.tools.Tool] = []
    for t in tools:
        if t.kind == "provider":
            builtin.append(t)
        else:
            custom.append(t)
    return custom, builtin


def _custom_tools_to_anthropic(
    tools: Sequence[types.tools.Tool],
) -> list[dict[str, Any]]:
    """Convert host-executed tools to Anthropic tool schema format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        args = tool.args
        if not isinstance(args, types.tools.FunctionToolArgs):
            raise TypeError(f"function tool {tool.name!r} has invalid args")
        result.append(
            {
                "name": tool.name,
                "description": args.description or "",
                "input_schema": args.params,
            }
        )
    return result


def _builtin_tools_to_anthropic(
    builtin: Sequence[types.tools.Tool],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Convert built-in tools to Anthropic wire format.

    Returns ``(wire_tools, beta_headers)``. Beta headers are merged into
    the ``anthropic-beta`` request header by the caller.

    Provider tool schemas keep args in the snake_case shape the native
    Anthropic API expects.
    """
    wire: list[dict[str, Any]] = []
    betas: set[str] = set()
    for tool in builtin:
        args_model = tool.args
        if not isinstance(args_model, anthropic_tools.AnthropicProviderArgs):
            raise ValueError(
                "AnthropicModel does not support provider args "
                f"{type(args_model).__name__}"
            )
        args = args_model.model_dump(mode="json", exclude_none=True)
        block: dict[str, Any] = {
            "type": args_model.anthropic_type,
            "name": tool.name,
            **args,
        }
        wire.append(block)
        if args_model.anthropic_beta is not None:
            betas.add(args_model.anthropic_beta)

    return wire, betas


def _file_part_to_anthropic(
    part: types.messages.FilePart,
) -> dict[str, Any]:
    """Convert a :class:`FilePart` to an Anthropic content block.

    * ``image/*`` -> ``{"type": "image", "source": ...}``
    * ``application/pdf`` -> ``{"type": "document", "source": ...}``
    * ``text/plain`` -> ``{"type": "document", "source": ...}``
    * anything else -> ``ValueError``
    """
    mt = part.media_type

    if mt.startswith("image/"):
        media_type = "image/jpeg" if mt == "image/*" else mt
        if isinstance(part.data, str) and types.media.is_url(part.data):
            return {
                "type": "image",
                "source": {"type": "url", "url": part.data},
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": types.media.data_to_base64(part.data),
            },
        }

    if mt == "application/pdf":
        if isinstance(part.data, str) and types.media.is_url(part.data):
            return {
                "type": "document",
                "source": {"type": "url", "url": part.data},
            }
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": types.media.data_to_base64(part.data),
            },
        }

    if mt == "text/plain":
        if isinstance(part.data, bytes):
            text_data = part.data.decode("utf-8")
        elif types.media.is_url(part.data):
            return {
                "type": "document",
                "source": {"type": "url", "url": part.data},
            }
        else:
            text_data = base64.b64decode(part.data).decode("utf-8")
        return {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": text_data,
            },
        }

    raise ValueError(f"Unsupported media type for Anthropic: {mt}")


def _tool_result_to_anthropic(value: Any) -> str | list[dict[str, Any]]:
    """Convert a tool result's model-facing value to Anthropic content.

    A :class:`ContentOutput` expands into Anthropic content blocks
    (image/document) so the model sees actual media.  Everything else is
    sent as a string (the Anthropic API accepts a string as tool_result
    content): ``str`` raw, ``None`` as ``""``, anything else JSON-encoded.
    """
    if isinstance(value, types.messages.ContentOutput):
        blocks: list[dict[str, Any]] = []
        for item in value.value:
            if isinstance(item, types.messages.FilePart):
                blocks.append(_file_part_to_anthropic(item))
            else:
                blocks.append({"type": "text", "text": item.text})
        return blocks
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), default=str)


async def _messages_to_anthropic(
    messages: list[types.messages.Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert internal messages to Anthropic API format.

    Returns ``(system_prompt, messages_list)``.  The system prompt is
    extracted separately because the Anthropic API takes it as a
    top-level parameter.
    """
    system_prompt: str | None = None
    result: list[dict[str, Any]] = []

    for msg in messages:
        match msg.role:
            case "system":
                system_prompt = "".join(
                    p.text
                    for p in msg.parts
                    if isinstance(p, types.messages.TextPart)
                )
            case "assistant":
                content: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.messages.ReasoningPart(
                            text=text,
                            provider_metadata=provider_metadata,
                        ):
                            signature = _anthropic_metadata(
                                provider_metadata
                            ).get("signature")
                            if signature:
                                content.append(
                                    {
                                        "type": "thinking",
                                        "thinking": text,
                                        "signature": signature,
                                    }
                                )
                        case types.messages.TextPart(text=text):
                            content.append({"type": "text", "text": text})
                        case types.messages.ToolCallPart():
                            tool_input = (
                                json.loads(part.tool_args)
                                if part.tool_args
                                else {}
                            )
                            content.append(
                                {
                                    "type": "tool_use",
                                    "id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "input": tool_input,
                                }
                            )
                        case types.messages.BuiltinToolCallPart():
                            btc_input = (
                                json.loads(part.tool_args)
                                if part.tool_args
                                else {}
                            )
                            content.append(
                                {
                                    "type": "server_tool_use",
                                    "id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "input": btc_input,
                                }
                            )
                        case types.messages.BuiltinToolReturnPart():
                            # Result block type comes from the original wire
                            # event ("web_search_tool_result", etc.); stored in
                            # provider metadata when emitted.
                            part_metadata = _anthropic_metadata(
                                part.provider_metadata
                            )
                            wire_type = (
                                part_metadata.get("resultType")
                                or f"{part.tool_name}_tool_result"
                            )
                            content.append(
                                {
                                    "type": wire_type,
                                    "tool_use_id": part.tool_call_id,
                                    "content": part.result,
                                }
                            )
                if content:
                    result.append({"role": "assistant", "content": content})

            case "tool":
                tool_results: list[dict[str, Any]] = []
                for part in msg.parts:
                    if isinstance(part, types.messages.ToolResultPart):
                        tool_content = _tool_result_to_anthropic(
                            part.get_model_input()
                        )
                        entry: dict[str, Any] = {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": tool_content,
                        }
                        if part.is_error:
                            entry["is_error"] = True
                        tool_results.append(entry)
                if tool_results:
                    result.append({"role": "user", "content": tool_results})

            case "user":
                has_files = any(
                    isinstance(p, types.messages.FilePart) for p in msg.parts
                )
                if not has_files:
                    content_text = "".join(
                        p.text
                        for p in msg.parts
                        if isinstance(p, types.messages.TextPart)
                    )
                    result.append({"role": "user", "content": content_text})
                else:
                    user_content: list[dict[str, Any]] = []
                    for p in msg.parts:
                        match p:
                            case types.messages.TextPart(text=text):
                                user_content.append(
                                    {"type": "text", "text": text}
                                )
                            case types.messages.FilePart():
                                user_content.append(_file_part_to_anthropic(p))
                    result.append({"role": "user", "content": user_content})

    result = _merge_consecutive_roles(result)
    return system_prompt, result


def _merge_consecutive_roles(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive messages that share the same role.

    Anthropic requires strictly alternating user/assistant roles.
    """
    if not messages:
        return messages

    merged: list[dict[str, Any]] = [messages[0]]

    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = _to_content_list(merged[-1]["content"])
            cur = _to_content_list(msg["content"])
            merged[-1]["content"] = prev + cur
        else:
            merged.append(msg)

    return merged


def _to_content_list(content: Any) -> list[dict[str, Any]]:
    """Normalize Anthropic message content to list-of-blocks."""
    if isinstance(content, list):
        return cast("list[dict[str, Any]]", list(content))
    return [{"type": "text", "text": content}]


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


def _extra_body(api_kwargs: dict[str, Any]) -> dict[str, Any]:
    extra_body = api_kwargs.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
        api_kwargs["extra_body"] = extra_body
    return extra_body


def _apply_output_config(
    api_kwargs: dict[str, Any],
    values: Mapping[str, Any],
) -> None:
    output_config = dict(api_kwargs.get("output_config") or {})
    output_config.update(values)
    api_kwargs["output_config"] = output_config


def _anthropic_tool_choice(
    tool_choice: params_.ToolChoiceMode
    | params_.ToolRef
    | params_.ToolSelection,
) -> dict[str, Any]:
    if isinstance(tool_choice, params_.ToolChoiceMode):
        match tool_choice:
            case params_.ToolChoiceMode.AUTO:
                return {"type": "auto"}
            case params_.ToolChoiceMode.REQUIRED:
                return {"type": "any"}
            case params_.ToolChoiceMode.NONE:
                return {"type": "none"}
    if isinstance(tool_choice, params_.ToolRef):
        return {"type": "tool", "name": str(tool_choice)}
    if len(tool_choice.tools) == 1 and tool_choice.mode in {
        params_.ToolChoiceMode.AUTO,
        params_.ToolChoiceMode.REQUIRED,
    }:
        return {"type": "tool", "name": str(next(iter(tool_choice.tools)))}
    raise ValueError("Anthropic does not support allowed tool subsets")


def _apply_sampling(
    api_kwargs: dict[str, Any],
    request_params: params_.InferenceRequestParams,
) -> None:
    sampling = request_params.sampling
    if isinstance(sampling, params_.ModelProviderDefault):
        return
    for sampler in sampling.values():
        match sampler:
            case params_.TemperatureSamplerParams(temperature=temperature):
                if _not_default(temperature):
                    api_kwargs["temperature"] = temperature
            case params_.TopPSamplerParams(top_p=top_p):
                if _not_default(top_p):
                    api_kwargs["top_p"] = top_p
            case params_.TopKSamplerParams(top_k=top_k):
                if _not_default(top_k):
                    if top_k is None:
                        raise ValueError("Anthropic top_k cannot be None")
                    api_kwargs["top_k"] = top_k
            case params_.SeedSamplerParams(seed=seed):
                if _not_default(seed) and _seed_value(seed) is not None:
                    raise ValueError("Anthropic does not support seed")
            case params_.MinPSamplerParams(min_p=min_p):
                if _not_default(min_p) and min_p is not None:
                    raise ValueError("Anthropic does not support min_p")
            case params_.RepetitionPenaltyParams() as repetition:
                unsupported = {
                    "repetition_penalty": repetition.repetition_penalty,
                    "frequency_penalty": repetition.frequency_penalty,
                    "presence_penalty": repetition.presence_penalty,
                    "consideration_window": repetition.consideration_window,
                }
                for key, value in unsupported.items():
                    if _not_default(value) and value is not None:
                        raise ValueError(f"Anthropic does not support {key}")


def _apply_anthropic_params(
    api_kwargs: dict[str, Any],
    request_params: params_.InferenceRequestParams,
    *,
    provider: str,
) -> None:
    _ = provider
    disable_parallel_tool_use = None
    _apply_sampling(api_kwargs, request_params)

    reasoning = request_params.reasoning
    output = request_params.output
    summary = params_.DEFAULT if output is None else output.reasoning_summary
    if not isinstance(reasoning, params_.ModelProviderDefault) and _not_default(
        reasoning.effort
    ):
        if reasoning.effort is None:
            api_kwargs["thinking"] = {"type": "disabled"}
        else:
            _apply_output_config(api_kwargs, {"effort": reasoning.effort})
    if _not_default(summary):
        if summary is None:
            api_kwargs["thinking"] = {"type": "disabled"}
        else:
            thinking = dict(api_kwargs.get("thinking") or {})
            thinking.setdefault("type", "adaptive")
            thinking["display"] = summary
            api_kwargs["thinking"] = thinking

    if request_params.tool_calling is not None:
        tool_calling = request_params.tool_calling
        if (
            _not_default(tool_calling.max_tool_calls)
            and tool_calling.max_tool_calls is not None
        ):
            raise ValueError("Anthropic does not support max_tool_calls")
        tool_choice = _anthropic_tool_choice(tool_calling.tool_choice)
        if _not_default(tool_calling.parallel_tool_calls):
            disable_parallel_tool_use = not tool_calling.parallel_tool_calls
        if disable_parallel_tool_use is not None:
            if tool_choice["type"] == "none":
                raise ValueError(
                    "Anthropic cannot set parallel tool calls with tool none"
                )
            tool_choice["disable_parallel_tool_use"] = disable_parallel_tool_use
        api_kwargs["tool_choice"] = tool_choice
    elif disable_parallel_tool_use is not None:
        api_kwargs["tool_choice"] = {
            "type": "auto",
            "disable_parallel_tool_use": disable_parallel_tool_use,
        }

    if request_params.provider_service is not None:
        service = request_params.provider_service
        if _not_default(service.service_tier):
            api_kwargs["service_tier"] = service.service_tier

    metadata: dict[str, str] = {}
    if request_params.metadata is not None:
        metadata.update(request_params.metadata)
    if request_params.safety_identifier is not None:
        metadata["user_id"] = request_params.safety_identifier
    if metadata:
        api_kwargs["metadata"] = metadata

    if request_params.context_management is not None:
        context_management = request_params.context_management
        if context_management.compaction is not None:
            _extra_body(api_kwargs)["context_management"] = {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": context_management.compaction.value,
                        },
                    }
                ]
            }

    if request_params.output is not None:
        output = request_params.output
        if output.max_tokens is not None:
            api_kwargs["max_tokens"] = output.max_tokens
        if output.include is not None:
            raise ValueError("Anthropic does not support output include")
        if (
            _not_default(output.text_verbosity)
            and output.text_verbosity is not None
        ):
            raise ValueError("Anthropic does not support text verbosity")

    if request_params.cache is not None:
        cache = request_params.cache
        if cache.key is not None:
            raise ValueError("Anthropic does not support cache keys")
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        if _not_default(cache.retention):
            cache_control["ttl"] = cache.retention
        api_kwargs["cache_control"] = cache_control

    extra_headers = _filter_extra_headers(request_params.extra_headers)
    if extra_headers is not None:
        api_kwargs["extra_headers"] = extra_headers
    if (
        request_params.context_management is not None
        and request_params.context_management.compaction is not None
    ):
        _add_builtin_beta_headers(
            api_kwargs,
            {"compact-2026-01-12", "context-management-2025-06-27"},
        )
    if request_params.extra_query is not None:
        api_kwargs["extra_query"] = dict(request_params.extra_query)
    if request_params.extra_body is not None:
        _extra_body(api_kwargs).update(request_params.extra_body)


def _coerce_params(
    value: params_.InferenceRequestParams | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, params_.InferenceRequestParams):
        api_kwargs: dict[str, Any] = {}
        _apply_anthropic_params(
            api_kwargs,
            value,
            provider=PROVIDER_NAME,
        )
        return api_kwargs
    raise TypeError("anthropic stream params must be InferenceRequestParams")


def _add_builtin_beta_headers(
    api_kwargs: dict[str, Any],
    betas: set[str],
) -> None:
    """Attach tool-required beta headers unless the caller supplied one."""
    if not betas:
        return
    headers = dict(api_kwargs.get("extra_headers") or {})
    if not any(key.lower() == "anthropic-beta" for key in headers):
        headers["anthropic-beta"] = ",".join(sorted(betas))
    api_kwargs["extra_headers"] = headers


def _result_block_content(block: Any) -> Any:
    """Serialize a server tool result block's content to JSON-friendly data."""
    content = getattr(block, "content", None)
    if content is None:
        return None
    if isinstance(content, pydantic.BaseModel):
        return content.model_dump(exclude_none=True)
    if isinstance(content, list):
        out: list[Any] = []
        for item in content:
            if isinstance(item, pydantic.BaseModel):
                out.append(item.model_dump(exclude_none=True))
            else:
                out.append(item)
        return out
    return content


# ---------------------------------------------------------------------------
# Public protocol function
# ---------------------------------------------------------------------------


async def stream(
    sdk_client: anthropic.AsyncAnthropic,
    model: core.model.Model,
    messages: list[types.messages.Message],
    *,
    tools: Sequence[types.tools.Tool] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    params: params_.InferenceRequestParams | None = None,
    provider: str,
) -> AsyncGenerator[events.Event]:
    """Stream through the Anthropic messages protocol using *sdk_client*.

    Yields :class:`~ai.types.events.Event` objects as the response streams in.
    Pure delta emitter — the :class:`~ai.models.Stream` wrapper aggregates
    parts into the final :class:`~ai.types.messages.Message`.

    ``params`` may be a raw dict of Anthropic SDK kwargs. Provider-specific
    request options are forwarded without local validation or translation.
    """
    anthropic_sdk = _sdk.import_sdk(provider=provider)
    stream_params = _coerce_params(params)
    if params is not None:
        stream_params = {}
        _apply_anthropic_params(stream_params, params, provider=provider)
    system_prompt, anthropic_messages = await _messages_to_anthropic(messages)

    custom_tools, builtin_tools = _split_tools(tools or ())
    wire_tools = (
        _custom_tools_to_anthropic(custom_tools) if custom_tools else []
    )
    builtin_betas: set[str] = set()
    if builtin_tools:
        builtin_wire, builtin_betas = _builtin_tools_to_anthropic(builtin_tools)
        wire_tools.extend(builtin_wire)

    api_kwargs: dict[str, Any] = dict(stream_params)
    api_kwargs.setdefault("max_tokens", 8192)
    api_kwargs.update(
        {
            "model": model.id,
            "messages": anthropic_messages,
        }
    )
    if system_prompt:
        api_kwargs["system"] = system_prompt
    if wire_tools:
        api_kwargs["tools"] = wire_tools

    _add_builtin_beta_headers(api_kwargs, builtin_betas)

    if output_type is not None:
        api_kwargs["output_format"] = output_type

    # Anthropic indexes content blocks by int; map to string block_ids.
    block_types: dict[int, str] = {}
    tool_ids: dict[int, str] = {}
    tool_names: dict[int, str] = {}
    signature_buffer: dict[int, str] = {}

    try:
        async with sdk_client.messages.stream(**api_kwargs) as sdk_stream:
            yield events.StreamStart()

            async for event in sdk_stream:
                match event.type:
                    case "content_block_start":
                        event = cast("Any", event)
                        block = event.content_block
                        idx = event.index
                        block_types[idx] = block.type

                        match block.type:
                            case "text":
                                yield events.TextStart(block_id=str(idx))
                            case "thinking":
                                yield events.ReasoningStart(block_id=str(idx))
                            case "tool_use":
                                tool_ids[idx] = block.id
                                tool_names[idx] = block.name
                                yield events.ToolStart(
                                    tool_call_id=block.id,
                                    tool_name=block.name,
                                )
                            case "server_tool_use":
                                tool_ids[idx] = block.id
                                tool_names[idx] = block.name
                                yield events.BuiltinToolStart(
                                    tool_call_id=block.id,
                                    tool_name=block.name,
                                    provider_metadata=_provider_metadata(),
                                )
                            # Result blocks (web_search_tool_result etc.) arrive
                            # complete; we emit on stop so we have full content.

                    case "content_block_delta":
                        event = cast("Any", event)
                        delta = event.delta
                        idx = event.index

                        match delta.type:
                            case "text_delta":
                                yield events.TextDelta(
                                    chunk=delta.text,
                                    block_id=str(idx),
                                )
                            case "thinking_delta":
                                yield events.ReasoningDelta(
                                    chunk=delta.thinking,
                                    block_id=str(idx),
                                )
                            case "signature_delta":
                                signature_buffer[idx] = (
                                    signature_buffer.get(idx, "")
                                    + delta.signature
                                )
                            case "input_json_delta":
                                tool_id = tool_ids.get(idx)
                                if not tool_id:
                                    continue
                                if block_types.get(idx) == "server_tool_use":
                                    yield events.BuiltinToolDelta(
                                        chunk=delta.partial_json,
                                        tool_call_id=tool_id,
                                    )
                                else:
                                    yield events.ToolDelta(
                                        chunk=delta.partial_json,
                                        tool_call_id=tool_id,
                                    )

                    case "content_block_stop":
                        event = cast("Any", event)
                        idx = event.index
                        block_type = block_types.get(idx)
                        if block_type == "text":
                            yield events.TextEnd(block_id=str(idx))
                        elif block_type == "thinking":
                            signature = signature_buffer.get(idx)
                            yield events.ReasoningEnd(
                                block_id=str(idx),
                                provider_metadata=(
                                    _provider_metadata(signature=signature)
                                    if signature is not None
                                    else None
                                ),
                            )
                        elif block_type == "tool_use":
                            tool_id = tool_ids.get(idx)
                            if tool_id:
                                yield events.ToolEnd(
                                    tool_call_id=tool_id,
                                    tool_call=types.messages.DUMMY_TOOL_CALL,
                                )
                        elif block_type == "server_tool_use":
                            tool_id = tool_ids.get(idx)
                            if tool_id:
                                yield events.BuiltinToolEnd(
                                    tool_call_id=tool_id,
                                    tool_call=types.messages.BuiltinToolCallPart(
                                        tool_call_id=tool_id,
                                        tool_name=tool_names.get(idx, ""),
                                        provider_metadata=_provider_metadata(),
                                    ),
                                )
                        elif block_type in _TOOL_RESULT_BLOCK_TYPES:
                            # Look up the matching server_tool_use (by
                            # tool_use_id) from the snapshot so we have
                            # the canonical tool name.
                            snap = sdk_stream.current_message_snapshot
                            result_block = (
                                snap.content[idx]
                                if idx < len(snap.content)
                                else None
                            )
                            if result_block is None:
                                continue
                            tool_use_id = (
                                getattr(result_block, "tool_use_id", None) or ""
                            )
                            content_payload = _result_block_content(
                                result_block
                            )
                            # Look up the corresponding server_tool_use
                            # block to recover the tool name.
                            tool_name = ""
                            for cb in snap.content:
                                if (
                                    getattr(cb, "type", None)
                                    == "server_tool_use"
                                    and getattr(cb, "id", None) == tool_use_id
                                ):
                                    tool_name = getattr(cb, "name", "") or ""
                                    break
                            yield events.BuiltinToolResult(
                                tool_call_id=tool_use_id,
                                result=types.messages.BuiltinToolReturnPart(
                                    tool_call_id=tool_use_id,
                                    tool_name=tool_name,
                                    result=content_payload,
                                    provider_metadata=_provider_metadata(
                                        resultType=block_type or ""
                                    ),
                                ),
                            )

            snapshot = sdk_stream.current_message_snapshot
            sdk_usage = snapshot.usage
            cache_read = getattr(sdk_usage, "cache_read_input_tokens", None)
            usage = types.usage.Usage(
                # We combine input_tokens and cache_read_input_tokens,
                # to match the behavior of other providers.
                input_tokens=(
                    (sdk_usage.input_tokens or 0) + (cache_read or 0)
                ),
                output_tokens=sdk_usage.output_tokens or 0,
                cache_read_tokens=cache_read,
                cache_write_tokens=getattr(
                    sdk_usage, "cache_creation_input_tokens", None
                ),
                raw=sdk_usage.model_dump(exclude_none=True) or None,
            )
            yield events.StreamEnd(usage=usage)
    except anthropic_sdk.AnthropicError as exc:
        raise errors.map_error(
            exc,
            provider=provider,
            model_id=model.id,
        ) from exc


class AnthropicMessagesProtocol(base.ProviderProtocol[Any]):
    """Anthropic Messages API protocol."""

    def stream(
        self,
        client: anthropic.AsyncAnthropic,
        model: core.model.Model,
        messages: list[types.messages.Message],
        *,
        tools: Sequence[types.tools.Tool] | None = None,
        output_type: type[pydantic.BaseModel] | None = None,
        params: params_.InferenceRequestParams | None = None,
        provider: str,
    ) -> AsyncGenerator[events.Event]:
        return stream(
            client,
            model,
            messages,
            tools=tools,
            output_type=output_type,
            params=params,
            provider=provider,
        )
