"""Anthropic adapter — messages API.

Message/tool conversion and streaming via the official ``anthropic`` SDK.
The SDK client is constructed from :class:`Client` params on each call.
"""

import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import anthropic
import pydantic

from ... import types
from ...types import events
from ...types import messages as messages_
from ...types import tools as tools_
from .. import core
from . import tools as anthropic_tools
from .params import (
    AnthropicContainer,
    AnthropicCustomSkill,
    AnthropicDisabledThinking,
    AnthropicParams,
    AnthropicProviderSkill,
)

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

# ---------------------------------------------------------------------------
# Message / tool conversion — internal types → Anthropic wire format
# ---------------------------------------------------------------------------


def _split_tools(
    tools: Sequence[types.ToolLike],
) -> tuple[list[types.ToolSchemaLike], list[tools_.BuiltinTool]]:
    """Split ``tools`` into custom (host-executed) and built-in (provider-executed)."""
    custom: list[types.ToolSchemaLike] = []
    builtin: list[tools_.BuiltinTool] = []
    for t in tools:
        if isinstance(t, tools_.BuiltinTool):
            builtin.append(t)
        else:
            custom.append(t)
    return custom, builtin


def _custom_tools_to_anthropic(
    tools: Sequence[types.ToolSchemaLike],
) -> list[dict[str, Any]]:
    """Convert custom (host-executed) Tool objects to Anthropic tool schema format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.param_schema,
        }
        for tool in tools
    ]


def _builtin_tools_to_anthropic(
    builtin: Sequence[tools_.BuiltinTool],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Convert built-in tools to Anthropic wire format.

    Returns ``(wire_tools, beta_headers)``. Beta headers are merged into
    the ``anthropic-beta`` request header by the caller.
    """
    wire: list[dict[str, Any]] = []
    betas: set[str] = set()
    for tool in builtin:
        if not isinstance(tool, anthropic_tools._AnthropicBuiltin):
            raise ValueError(
                f"AnthropicModel does not support built-in tool {type(tool).__name__}"
            )
        block: dict[str, Any] = {
            "type": tool.wire_type,
            "name": tool.wire_name,
        }
        match tool:
            case anthropic_tools.WebSearch():
                if tool.max_uses is not None:
                    block["max_uses"] = tool.max_uses
                if tool.allowed_domains is not None:
                    block["allowed_domains"] = list(tool.allowed_domains)
                if tool.blocked_domains is not None:
                    block["blocked_domains"] = list(tool.blocked_domains)
                if tool.user_location is not None:
                    block["user_location"] = {
                        "type": "approximate",
                        **tool.user_location,
                    }
            case anthropic_tools.WebFetch():
                if tool.max_uses is not None:
                    block["max_uses"] = tool.max_uses
                if tool.allowed_domains is not None:
                    block["allowed_domains"] = list(tool.allowed_domains)
                if tool.blocked_domains is not None:
                    block["blocked_domains"] = list(tool.blocked_domains)
                if tool.citations is not None:
                    block["citations"] = tool.citations.model_dump()
                if tool.max_content_tokens is not None:
                    block["max_content_tokens"] = tool.max_content_tokens
            case anthropic_tools.ComputerUse():
                block["display_width_px"] = tool.display_width_px
                block["display_height_px"] = tool.display_height_px
                if tool.display_number is not None:
                    block["display_number"] = tool.display_number
                if tool.enable_zoom is not None:
                    block["enable_zoom"] = tool.enable_zoom
            case anthropic_tools.TextEditor():
                if tool.max_characters is not None:
                    block["max_characters"] = tool.max_characters
            case (
                anthropic_tools.CodeExecution()
                | anthropic_tools.Bash()
                | anthropic_tools.Memory()
            ):
                pass
            case _:
                raise ValueError(
                    f"AnthropicModel does not support built-in tool "
                    f"{type(tool).__name__}"
                )

        wire.append(block)
        if tool.beta is not None:
            betas.add(tool.beta)

    return wire, betas


def _file_part_to_anthropic(
    part: types.FilePart,
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
            import base64 as _b64

            text_data = _b64.b64decode(part.data).decode("utf-8")
        return {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": text_data,
            },
        }

    raise ValueError(f"Unsupported media type for Anthropic: {mt}")


async def _messages_to_anthropic(
    messages: list[types.Message],
    *,
    send_reasoning: bool = True,
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
                    p.text for p in msg.parts if isinstance(p, types.TextPart)
                )
            case "assistant":
                content: list[dict[str, Any]] = []
                for part in msg.parts:
                    match part:
                        case types.ReasoningPart(text=text, signature=signature):
                            if send_reasoning and signature:
                                content.append(
                                    {
                                        "type": "thinking",
                                        "thinking": text,
                                        "signature": signature,
                                    }
                                )
                        case types.TextPart(text=text):
                            content.append({"type": "text", "text": text})
                        case types.ToolCallPart():
                            tool_input = (
                                json.loads(part.tool_args) if part.tool_args else {}
                            )
                            content.append(
                                {
                                    "type": "tool_use",
                                    "id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "input": tool_input,
                                }
                            )
                        case types.BuiltinToolCallPart():
                            btc_input = (
                                json.loads(part.tool_args) if part.tool_args else {}
                            )
                            content.append(
                                {
                                    "type": "server_tool_use",
                                    "id": part.tool_call_id,
                                    "name": part.tool_name,
                                    "input": btc_input,
                                }
                            )
                        case types.BuiltinToolReturnPart():
                            # Result block type comes from the original wire
                            # event ("web_search_tool_result", etc.); stored
                            # in provider_details when emitted.
                            details = part.provider_details or {}
                            wire_type = details.get(
                                "result_type",
                                f"{part.tool_name}_tool_result",
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
                    if isinstance(part, types.ToolResultPart):
                        entry: dict[str, Any] = {
                            "type": "tool_result",
                            "tool_use_id": part.tool_call_id,
                            "content": str(part.result)
                            if part.result is not None
                            else "",
                        }
                        if part.is_error:
                            entry["is_error"] = True
                        tool_results.append(entry)
                if tool_results:
                    result.append({"role": "user", "content": tool_results})

            case "user":
                has_files = any(isinstance(p, types.FilePart) for p in msg.parts)
                if not has_files:
                    content_text = "".join(
                        p.text for p in msg.parts if isinstance(p, types.TextPart)
                    )
                    result.append({"role": "user", "content": content_text})
                else:
                    user_content: list[dict[str, Any]] = []
                    for p in msg.parts:
                        match p:
                            case types.TextPart(text=text):
                                user_content.append({"type": "text", "text": text})
                            case types.FilePart():
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
        return list(content)
    return [{"type": "text", "text": content}]


# ---------------------------------------------------------------------------
# SDK client factory
# ---------------------------------------------------------------------------


def _make_client(
    client: core.client.Client,
) -> anthropic.AsyncAnthropic:
    """Construct an ``AsyncAnthropic`` from our generic ``Client``."""
    return anthropic.AsyncAnthropic(
        base_url=client.base_url,
        api_key=client.api_key or "",
    )


def _coerce_anthropic_params(value: Any) -> AnthropicParams:
    if value is None:
        return AnthropicParams()
    if isinstance(value, AnthropicParams):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = list(value)
        if len(items) == 1 and isinstance(items[0], AnthropicParams):
            return items[0]
    raise TypeError(f"anthropic stream params must be {AnthropicParams.__name__}")


def _container_to_wire(container: AnthropicContainer) -> str | dict[str, Any] | None:
    if not container.skills:
        return container.id

    result: dict[str, Any] = {}
    if container.id is not None:
        result["id"] = container.id
    result["skills"] = [_skill_to_wire(skill) for skill in container.skills]
    return result


def _skill_to_wire(
    skill: AnthropicProviderSkill | AnthropicCustomSkill,
) -> dict[str, Any]:
    if isinstance(skill, AnthropicProviderSkill):
        result: dict[str, Any] = {
            "type": "anthropic",
            "skill_id": skill.skill_id,
        }
    else:
        result = {
            "type": "custom",
            "provider_reference": skill.provider_reference,
        }
    if skill.version is not None:
        result["version"] = skill.version
    return result


def _merge_extra_headers(
    api_kwargs: dict[str, Any],
    extra_headers: dict[str, str] | None,
) -> None:
    """Attach raw headers while preserving typed headers the adapter generated."""
    headers = dict(api_kwargs.get("extra_headers") or {})
    if extra_headers:
        headers.update(extra_headers)
    if headers:
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
# Public adapter function
# ---------------------------------------------------------------------------


async def stream(
    client: core.client.Client,
    model: core.model.Model[Any],
    messages: list[types.Message],
    *,
    tools: Sequence[types.ToolLike] | None = None,
    output_type: type[pydantic.BaseModel] | None = None,
    thinking: bool = False,
    budget_tokens: int = 10000,
    **kwargs: Any,
) -> AsyncGenerator[events.Event]:
    """Stream an LLM response via the Anthropic messages API.

    Yields :class:`~ai.types.events.Event` objects as the response streams in.
    Pure delta emitter — the :class:`~ai.models.Stream` wrapper aggregates
    parts into the final :class:`~ai.types.Message`.

    Extra keyword arguments beyond the ``StreamFn`` protocol:

    * ``thinking`` — enable extended thinking output.
    * ``budget_tokens`` — max tokens for thinking (default 10000).
    """
    sdk_client = _make_client(client)
    anthropic_params = _coerce_anthropic_params(kwargs.get("params"))
    system_prompt, anthropic_messages = await _messages_to_anthropic(
        messages,
        send_reasoning=anthropic_params.send_reasoning is not False,
    )

    custom_tools, builtin_tools = _split_tools(tools or ())
    wire_tools = _custom_tools_to_anthropic(custom_tools) if custom_tools else []
    builtin_betas: set[str] = set()
    if builtin_tools:
        builtin_wire, builtin_betas = _builtin_tools_to_anthropic(builtin_tools)
        wire_tools.extend(builtin_wire)

    api_kwargs: dict[str, Any] = {
        "model": model.id,
        "messages": anthropic_messages,
        "max_tokens": 8192,
    }
    if system_prompt:
        api_kwargs["system"] = system_prompt
    if wire_tools:
        api_kwargs["tools"] = wire_tools

    if anthropic_params.thinking is not None:
        if not isinstance(anthropic_params.thinking, AnthropicDisabledThinking):
            api_kwargs["thinking"] = anthropic_params.thinking.model_dump(
                exclude_none=True,
            )
    elif thinking:
        api_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }

    output_config: dict[str, Any] = {}
    if anthropic_params.effort is not None:
        output_config["effort"] = anthropic_params.effort
    if anthropic_params.task_budget is not None:
        output_config["task_budget"] = anthropic_params.task_budget.model_dump(
            exclude_none=True,
        )
    if output_config:
        api_kwargs["output_config"] = output_config

    if anthropic_params.speed is not None:
        api_kwargs["speed"] = anthropic_params.speed
    if anthropic_params.inference_geo is not None:
        api_kwargs["inference_geo"] = anthropic_params.inference_geo
    if anthropic_params.metadata is not None:
        api_kwargs["metadata"] = anthropic_params.metadata.model_dump(
            exclude_none=True,
        )
    if anthropic_params.context_management is not None:
        api_kwargs["context_management"] = anthropic_params.context_management
    if anthropic_params.container is not None:
        container = _container_to_wire(anthropic_params.container)
        if container is not None:
            api_kwargs["container"] = container
    if anthropic_params.service_tier is not None:
        api_kwargs["service_tier"] = anthropic_params.service_tier
    if anthropic_params.cache_control is not None:
        api_kwargs["cache_control"] = anthropic_params.cache_control.model_dump(
            exclude_none=True,
        )
    if anthropic_params.mcp_servers is not None:
        api_kwargs["mcp_servers"] = [
            server.model_dump(exclude_none=True)
            for server in anthropic_params.mcp_servers
        ]
    if anthropic_params.disable_parallel_tool_use is not None:
        api_kwargs["tool_choice"] = {
            "type": "auto",
            "disable_parallel_tool_use": anthropic_params.disable_parallel_tool_use,
        }
    merged_betas: list[str] = []
    seen_betas: set[str] = set()
    for beta in list(anthropic_params.betas or ()) + sorted(builtin_betas):
        if beta not in seen_betas:
            seen_betas.add(beta)
            merged_betas.append(beta)
    if merged_betas:
        api_kwargs["extra_headers"] = {"anthropic-beta": ",".join(merged_betas)}
    _merge_extra_headers(api_kwargs, anthropic_params.extra_headers)
    if anthropic_params.extra_body:
        api_kwargs["extra_body"] = dict(anthropic_params.extra_body)

    if (
        output_type is not None
        and anthropic_params.structured_output_mode != "json_tool"
    ):
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
                                    provider_name=PROVIDER_NAME,
                                )
                            # Result blocks (web_search_tool_result etc.) arrive
                            # complete; we emit on stop so we have full content.

                    case "content_block_delta":
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
                                    signature_buffer.get(idx, "") + delta.signature
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
                        idx = event.index
                        block_type = block_types.get(idx)
                        if block_type == "text":
                            yield events.TextEnd(block_id=str(idx))
                        elif block_type == "thinking":
                            yield events.ReasoningEnd(
                                block_id=str(idx),
                                signature=signature_buffer.get(idx),
                            )
                        elif block_type == "tool_use":
                            tool_id = tool_ids.get(idx)
                            if tool_id:
                                yield events.ToolEnd(
                                    tool_call_id=tool_id,
                                    tool_call=messages_.DUMMY_TOOL_CALL,
                                )
                        elif block_type == "server_tool_use":
                            tool_id = tool_ids.get(idx)
                            if tool_id:
                                yield events.BuiltinToolEnd(
                                    tool_call_id=tool_id,
                                    tool_call=messages_.BuiltinToolCallPart(
                                        tool_call_id=tool_id,
                                        tool_name=tool_names.get(idx, ""),
                                        provider_name=PROVIDER_NAME,
                                    ),
                                )
                        elif block_type in _TOOL_RESULT_BLOCK_TYPES:
                            # Look up the matching server_tool_use (by
                            # tool_use_id) from the snapshot so we have
                            # the canonical tool name.
                            snap = sdk_stream.current_message_snapshot
                            result_block = (
                                snap.content[idx] if idx < len(snap.content) else None
                            )
                            if result_block is None:
                                continue
                            tool_use_id = (
                                getattr(result_block, "tool_use_id", None) or ""
                            )
                            content_payload = _result_block_content(result_block)
                            # Look up the corresponding server_tool_use
                            # block to recover the tool name.
                            tool_name = ""
                            for cb in snap.content:
                                if (
                                    getattr(cb, "type", None) == "server_tool_use"
                                    and getattr(cb, "id", None) == tool_use_id
                                ):
                                    tool_name = getattr(cb, "name", "") or ""
                                    break
                            yield events.BuiltinToolResult(
                                tool_call_id=tool_use_id,
                                result=messages_.BuiltinToolReturnPart(
                                    tool_call_id=tool_use_id,
                                    tool_name=tool_name,
                                    result=content_payload,
                                    provider_name=PROVIDER_NAME,
                                    provider_details={
                                        "result_type": block_type or "",
                                    },
                                ),
                            )

            snapshot = sdk_stream.current_message_snapshot
            sdk_usage = snapshot.usage
            usage = types.Usage(
                input_tokens=sdk_usage.input_tokens or 0,
                output_tokens=sdk_usage.output_tokens or 0,
                cache_read_tokens=getattr(sdk_usage, "cache_read_input_tokens", None),
                cache_write_tokens=getattr(
                    sdk_usage, "cache_creation_input_tokens", None
                ),
                raw=sdk_usage.model_dump(exclude_none=True) or None,
            )
            yield events.StreamEnd(usage=usage)
    finally:
        await sdk_client.close()
