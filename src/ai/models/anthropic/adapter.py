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
from .. import core

# ---------------------------------------------------------------------------
# Message / tool conversion — internal types → Anthropic wire format
# ---------------------------------------------------------------------------


def _tools_to_anthropic(
    tools: Sequence[types.ToolLike],
) -> list[dict[str, Any]]:
    """Convert internal Tool objects to Anthropic tool schema format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.param_schema,
        }
        for tool in tools
    ]


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
                            if signature:
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


# ---------------------------------------------------------------------------
# Public adapter function
# ---------------------------------------------------------------------------


async def stream(
    client: core.client.Client,
    model: core.model.Model,
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
    system_prompt, anthropic_messages = await _messages_to_anthropic(messages)
    anthropic_tools = _tools_to_anthropic(tools) if tools else None

    api_kwargs: dict[str, Any] = {
        "model": model.id,
        "messages": anthropic_messages,
        "max_tokens": 8192,
    }
    if system_prompt:
        api_kwargs["system"] = system_prompt
    if anthropic_tools:
        api_kwargs["tools"] = anthropic_tools

    if thinking:
        api_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }

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
                                if tool_id:
                                    yield events.ToolDelta(
                                        chunk=delta.partial_json,
                                        tool_call_id=tool_id,
                                    )

                    case "content_block_stop":
                        idx = event.index
                        match block_types.get(idx):
                            case "text":
                                yield events.TextEnd(block_id=str(idx))
                            case "thinking":
                                yield events.ReasoningEnd(
                                    block_id=str(idx),
                                    signature=signature_buffer.get(idx),
                                )
                            case "tool_use":
                                tool_id = tool_ids.get(idx)
                                if tool_id:
                                    yield events.ToolEnd(tool_call_id=tool_id)

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
